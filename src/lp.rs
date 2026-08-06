use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::check_pos;

// Linear programme in one standard form, `A` dense and small:
//
//      min  c . v    subject to   A v = b,   v >= 0
//
// Consumers: Curty's MDI-BB84 decoy grid, mode pairing's pair photon-number matrix, Yin's
// multiphoton RRDPS bound. Same scheme as `sdp.rs` and `herm.rs`, over the DIAGONAL cone.
//
// THE METHOD IS THE SECURITY ARGUMENT: log-barrier path-following on the DUAL,
//
//      max  b . y    subject to   z = c - A^T y  >=  0,
//
// so by weak duality `b . y` at ANY iterate lower-bounds the primal minimum. An early stop, stall
// or breakdown returns a LOOSER bound, never a wrong one.
//
// Consumers arrange the sign so low is pessimistic: a yield is `min Y_11`; an error or an
// information (`max e_11`, `max I_AE`) is `min` of its negative.
//
// ASSEMBLY WARNING: never state one pair of inequalities as two rows with identical coefficients.
// The dual gains a null direction, `|y|` runs to ~1e19, and round-off in `A v = b` passes the
// objective -- measured returning a certified 8.38 for a yield whose true value is 1.05e-2, every
// dual slack positive. Use one row with a bounded slack; `mdi::deck` is the worked form.
//
// `polytope_cap` bounds `max f` for CONCAVE `f` over `sum x = 1, x >= 0` plus `Facets`, given
// tangent planes `f(x) <= a_k + g_k . x`. Cut VALIDITY is a theorem; cut PLACEMENT moves only
// tightness. An assembly may drop a facet row (it only raises the maximum), never a cut.
//
// DUPLICATED, NOT SHARED: `chol`, `chol_solve` are byte-identical to `sdp::chol`,
// `sdp::chol_solve`, `herm::real_chol`, `herm::real_solve`; a `diff` of the bodies is the drift
// check.

/// Variables accepted. The widest RRDPS packet `rrdps::L_MAX` allows needs 4101.
const VAR_MAX: usize = 8192;

/// Equality rows accepted; a Newton step costs `m^2 n`. `rrdps::POLYTOPE_MAX` reads it.
pub(crate) const ROW_MAX: usize = 256;

// The six barrier constants differ from `sdp.rs`'s and `herm.rs`'s on purpose and trade only
// tightness against cost. Do NOT unify them; `RETRY_MAX` = 6 in all three means a different floor.

/// Newton steps per outer (barrier) iteration before the round is abandoned.
const NEWTON_MAX: usize = 100;

/// Squared Newton decrement at which a barrier subproblem counts as centred.
const CENTRED: f64 = 1e-11;

/// Barrier parameter growth per outer iteration.
const T_GROW: f64 = 12.0;

/// Backtracking floor. Below this the step is shorter than the f64 resolution of `y`.
const STEP_MIN: f64 = 1e-13;

/// Armijo constant for the barrier line search.
const ARMIJO: f64 = 0.25;

/// Retries allowed after a Newton round breaks down. Each halves the exponent of the
/// barrier step.
const RETRY_MAX: usize = 6;

/// Ridges tried, in order, on a Hessian whose Cholesky fails; relative, the scaled diagonal being 1.
const RIDGES: [f64; 4] = [0.0, 1e-13, 1e-11, 1e-9];

/// Retractions toward the strictly feasible start; each halves the distance, the last lands on it.
const REPAIR_MAX: usize = 64;

/// Relative slack between `polytope_cap`'s certificate and its cuts' floor. Not a barrier
/// constant: past it the polytope is empty, not the path loose.
const FLOOR_TOL: f64 = 1e-9;

/// Lower Cholesky factor of a symmetric `n x n` matrix, row-major. `None` when the matrix
/// is not positive definite.
fn chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];

            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }

            if i == j {
                if !(s > 0.0) {
                    return None;
                }

                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }

    Some(l)
}

/// Solve `A x = b` from the Cholesky factor of `A`, forward then back.
fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; n];

    for i in 0..n {
        let mut s = b[i];

        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }

        y[i] = s / l[i * n + i];
    }

    let mut x = vec![0.0f64; n];

    for step in 0..n {
        let i = n - 1 - step;
        let mut s = y[i];

        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }

        x[i] = s / l[i * n + i];
    }

    x
}

/// `min c . v` subject to `A v = b`, `v >= 0`. `a` is row-major `m x n`.
struct Lp {
    n: usize,
    m: usize,
    c: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
}

/// Best dual point, central-path gap `n/t` there, and why the run stopped. Once `certify` has
/// passed over it, `y` is dual feasible whatever `status` says.
struct Report {
    y: Vec<f64>,
    gap: f64,
    iters: usize,
    status: &'static str,
}

impl Lp {
    /// The dual slack `z = c - A^T y`, one entry per primal variable.
    fn slack(&self, y: &[f64]) -> Vec<f64> {
        let mut z = self.c.to_vec();

        for i in 0..self.m {
            let yi = y[i];

            if yi == 0.0 {
                continue;
            }

            let row = &self.a[i * self.n..(i + 1) * self.n];

            for (zj, aij) in z.iter_mut().zip(row.iter()) {
                *zj -= yi * aij;
            }
        }

        z
    }

    /// `b . y`, the dual objective.
    fn value(&self, y: &[f64]) -> f64 {
        let mut acc = 0.0;

        for (bi, yi) in self.b.iter().zip(y.iter()) {
            acc += bi * yi;
        }

        acc
    }

    /// Barrier value `-t (b . y) - sum_j ln z_j`, or `None` outside the open cone.
    fn barrier(&self, y: &[f64], t: f64) -> Option<f64> {
        let z = self.slack(y);
        let mut acc = -t * self.value(y);

        for zj in z.iter() {
            if !(*zj > 0.0) {
                return None;
            }

            acc -= zj.ln();
        }

        if acc.is_finite() {
            Some(acc)
        } else {
            None
        }
    }

    /// Newton direction and squared decrement, or `None` if the point or Hessian is unusable.
    /// Gradient `-t b + A z^-1`, Hessian `A diag(z^-2) A^T`, both exact.
    fn newton(&self, y: &[f64], t: f64) -> Option<(Vec<f64>, f64)> {
        let n = self.n;
        let m = self.m;
        let z = self.slack(y);
        let mut w = vec![0.0f64; m * n];
        let mut grad = vec![0.0f64; m];

        for j in 0..n {
            if !(z[j] > 0.0) {
                return None;
            }
        }

        for i in 0..m {
            let mut acc = -t * self.b[i];

            // `w`: the gradient sums it, the Hessian is its Gram matrix.
            for j in 0..n {
                let scaled = self.a[i * n + j] / z[j];
                w[i * n + j] = scaled;
                acc += scaled;
            }

            grad[i] = acc;
        }

        let mut hess = vec![0.0f64; m * m];

        for i in 0..m {
            for k in i..m {
                let mut acc = 0.0;

                for j in 0..n {
                    acc += w[i * n + j] * w[k * n + j];
                }

                hess[i * m + k] = acc;
                hess[k * m + i] = acc;
            }
        }

        // Jacobi scaling: cut and simplex rows differ by orders of magnitude.
        let mut d = vec![0.0f64; m];

        for i in 0..m {
            let diag = hess[i * m + i];

            if !(diag > 0.0) {
                return None;
            }

            d[i] = 1.0 / diag.sqrt();
        }

        let mut scaled = vec![0.0f64; m * m];

        for i in 0..m {
            for k in 0..m {
                scaled[i * m + k] = hess[i * m + k] * d[i] * d[k];
            }
        }

        let mut rhs = vec![0.0f64; m];

        for i in 0..m {
            rhs[i] = -grad[i] * d[i];
        }

        for ridge in RIDGES.iter() {
            let mut trial = scaled.to_vec();

            for i in 0..m {
                trial[i * m + i] += ridge;
            }

            if let Some(l) = chol(&trial, m) {
                let u = chol_solve(&l, m, &rhs);
                let mut dir = vec![0.0f64; m];
                let mut lam2 = 0.0;

                for i in 0..m {
                    dir[i] = u[i] * d[i];
                    lam2 -= grad[i] * dir[i];
                }

                if !dir.iter().all(|x| x.is_finite()) {
                    return None;
                }

                return Some((dir, lam2));
            }
        }

        None
    }

    /// Backtracking line search that never leaves the open cone.
    fn advance(&self, y: &[f64], dir: &[f64], t: f64, lam2: f64) -> Option<Vec<f64>> {
        let base = self.barrier(y, t)?;
        let mut step = 1.0f64;

        while step > STEP_MIN {
            let mut trial = vec![0.0f64; self.m];

            for i in 0..self.m {
                trial[i] = y[i] + step * dir[i];
            }

            if let Some(val) = self.barrier(&trial, t) {
                if val <= base - ARMIJO * step * lam2 {
                    return Some(trial);
                }
            }

            step *= 0.5;
        }

        None
    }

    /// Path-following from a strictly feasible `start` until the central-path gap `n/t`
    /// falls to `gaptol`. Returns the point from the last completed round.
    fn run(&self, start: &[f64], gaptol: f64) -> Report {
        let nu = self.n as f64;
        let mut y = start.to_vec();
        let mut best = start.to_vec();
        let mut grow = T_GROW;
        let mut held = 1.0f64;
        let mut t = 1.0f64;
        let mut iters = 0usize;
        let mut tries = 0usize;
        let mut status = "converged";

        loop {
            let mut broke = false;

            for _ in 0..NEWTON_MAX {
                iters += 1;

                match self.newton(&y, t) {
                    None => {
                        broke = true;
                        break;
                    }
                    Some((dir, lam2)) => {
                        if lam2 <= CENTRED {
                            break;
                        }

                        match self.advance(&y, &dir, t, lam2) {
                            None => {
                                broke = true;
                                break;
                            }
                            Some(next) => y = next,
                        }
                    }
                }
            }

            // Breakdown: retry from the last centred point with a square-rooted step.
            if broke {
                if tries >= RETRY_MAX || grow <= 1.0 + 1e-3 {
                    status = "breakdown";
                    break;
                }

                tries += 1;
                grow = grow.sqrt();
                y.copy_from_slice(&best);
                t = held * grow;
                continue;
            }

            best.copy_from_slice(&y);
            held = t;

            if nu / t <= gaptol {
                break;
            }

            t *= grow;

            if !t.is_finite() {
                status = "exhausted";
                break;
            }
        }

        Report {
            y: best,
            gap: nu / held,
            iters,
            status,
        }
    }

    /// Retract `y` toward `start` until `c - A^T y >= 0` holds in the REPORTED arithmetic, not
    /// the line search's. `z` is affine in `y`: one fraction clears every component.
    fn certify(&self, y: &[f64], start: &[f64]) -> Option<Vec<f64>> {
        let z = self.slack(y);
        let mut lowest = f64::INFINITY;

        for zj in z.iter() {
            lowest = lowest.min(*zj);
        }

        if lowest >= 0.0 {
            return Some(y.to_vec());
        }

        let z0 = self.slack(start);
        let mut share = 0.0f64;

        for (zj, z0j) in z.iter().zip(z0.iter()) {
            if *zj < 0.0 {
                let span = z0j - zj;

                if !(span > 0.0) {
                    return None;
                }

                share = share.max(-zj / span);
            }
        }

        for _ in 0..REPAIR_MAX {
            share = (share + 1e-15).min(1.0);

            let mut trial = vec![0.0f64; self.m];

            for i in 0..self.m {
                trial[i] = (1.0 - share) * y[i] + share * start[i];
            }

            let mut worst = f64::INFINITY;

            for zj in self.slack(&trial).iter() {
                worst = worst.min(*zj);
            }

            if worst >= 0.0 {
                return Some(trial);
            }

            share = 0.5 * (share + 1.0);
        }

        None
    }
}

/// `(value, gap, iters, status, y)`, as `solve` returns it.
type Solved = (f64, f64, usize, &'static str, Vec<f64>);

/// Run the path from `start` and certify its stop; `value = b . y` lower-bounds `min c . v`.
fn solve(lp: &Lp, start: &[f64], gaptol: f64) -> PyResult<Solved> {
    for zj in lp.slack(start).iter() {
        if !(*zj > 0.0) {
            return Err(PyValueError::new_err(
                "start is not strictly feasible for the dual: some entry of c - A^T start is \
                 <= 0; choose a start with c - A^T start > 0 in every entry",
            ));
        }
    }

    let rep = lp.run(start, gaptol);

    let y = match lp.certify(&rep.y, start) {
        Some(fixed) => fixed,
        None => {
            return Err(PyValueError::new_err(format!(
                "the dual point could not be certified after {} iterations with status \
                 \"{}\": it left the cone and retraction toward the start did not return it",
                rep.iters, rep.status
            )))
        }
    };

    let value = lp.value(&y);

    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "the dual objective is not finite at the certified point after {} iterations \
             with status \"{}\"",
            rep.iters, rep.status
        )));
    }

    Ok((value, rep.gap, rep.iters, rep.status, y))
}

/// Affine rows beyond `sum x = 1`: `eq . x = eqb` and `le . x <= leb`, one row per entry of
/// `eqb` and `leb`, matrices row-major over `dim` coordinates. All empty is the bare simplex.
pub(crate) struct Facets<'a> {
    pub(crate) eq: &'a [f64],
    pub(crate) eqb: &'a [f64],
    pub(crate) le: &'a [f64],
    pub(crate) leb: &'a [f64],
}

/// Certified upper bound on `max f` over the simplex cut down by `facets`, concave `f` given as
/// tangent planes `f(x) <= levels[k] + cuts[k] . x`, `cuts` row-major with `dim` entries per cut.
/// Returns `(bound, gap, iters, status)`.
///
/// With `tau` the epigraph variable, `w` the cut slacks, `s` one slack per inequality facet:
///
/// `min -tau  s.t.  g_k . x - tau - w_k = -(levels[k] + shift),  sum_i x_i = 1,`
/// `eq . x = eqb,  le . x + s = leb,  (x, tau, w, s) >= 0`,
///
/// with minimum `-(shift + max_x min_k (levels[k] + g_k . x))`.
///
/// `tau >= 0`, not free (a free variable leaves the dual cone no interior). Where the model's
/// maximum is negative that clamp empties the primal and the path reports a bound BELOW the
/// maximum; `shift` lifts the model's floor to zero to prevent it and comes off on return.
pub(crate) fn polytope_cap(
    dim: usize,
    cuts: &[f64],
    levels: &[f64],
    facets: &Facets,
    gaptol: f64,
) -> PyResult<(f64, f64, usize, &'static str)> {
    let k = levels.len();
    let e = facets.eqb.len();
    let f = facets.leb.len();

    if dim == 0 || k == 0 || cuts.len() != k * dim {
        return Err(PyValueError::new_err(format!(
            "a cutting-plane model needs at least one cut over at least one coordinate, \
             and one gradient entry per coordinate per cut: got {k} cuts, {dim} \
             coordinates and {} gradient entries",
            cuts.len()
        )));
    }

    if facets.eq.len() != e * dim || facets.le.len() != f * dim {
        return Err(PyValueError::new_err(format!(
            "a facet matrix carries one coefficient per coordinate per row: got {e} \
             equality rows against {} coefficients and {f} inequality rows against {}, \
             over {dim} coordinates",
            facets.eq.len(),
            facets.le.len()
        )));
    }

    let n = dim + 1 + k + f;
    let m = k + 1 + e + f;

    if n > VAR_MAX || m > ROW_MAX {
        return Err(PyValueError::new_err(format!(
            "this solver is written for small dense programmes: got {n} variables against \
             a cap of {VAR_MAX} and {m} rows against a cap of {ROW_MAX}"
        )));
    }

    // Each plane at its smallest coordinate floors the model over the simplex, hence the polytope.
    let mut floor = f64::INFINITY;

    for row in 0..k {
        let mut least = f64::INFINITY;

        for j in 0..dim {
            least = least.min(cuts[row * dim + j]);
        }

        floor = floor.min(levels[row] + least);
    }

    let shift = if floor < 0.0 { -floor } else { 0.0 };
    let mut c = vec![0.0f64; n];
    let mut a = vec![0.0f64; m * n];
    let mut b = vec![0.0f64; m];
    c[dim] = -1.0;

    for row in 0..k {
        let at = row * n;

        for j in 0..dim {
            a[at + j] = cuts[row * dim + j];
        }

        a[at + dim] = -1.0;
        a[at + dim + 1 + row] = -1.0;
        b[row] = -levels[row] - shift;
    }

    for j in 0..dim {
        a[k * n + j] = 1.0;
    }

    b[k] = 1.0;

    for row in 0..e {
        let at = (k + 1 + row) * n;

        for j in 0..dim {
            a[at + j] = facets.eq[row * dim + j];
        }

        b[k + 1 + row] = facets.eqb[row];
    }

    for row in 0..f {
        let at = (k + 1 + e + row) * n;

        for j in 0..dim {
            a[at + j] = facets.le[row * dim + j];
        }

        a[at + dim + 1 + k + row] = 1.0;
        b[k + 1 + e + row] = facets.leb[row];
    }

    let lp = Lp {
        n,
        m,
        c,
        a,
        b,
    };
    let share = 2.0 / (k as f64);
    let mut start = vec![0.0f64; m];
    let mut highest = f64::NEG_INFINITY;

    for si in start.iter_mut().take(k) {
        *si = share;
    }

    // Inequality multipliers start at -1, where `z_s = -y > 0`; equality multipliers at 0.
    for si in start.iter_mut().skip(k + 1 + e) {
        *si = -1.0;
    }

    for j in 0..dim {
        let mut col = 0.0;

        for row in 0..k {
            col += share * cuts[row * dim + j];
        }

        for row in 0..f {
            col -= facets.le[row * dim + j];
        }

        highest = highest.max(col);
    }

    if !highest.is_finite() {
        return Err(PyValueError::new_err(
            "a cut gradient is not finite, so the cutting-plane model has no interior",
        ));
    }

    if !floor.is_finite() {
        return Err(PyValueError::new_err(
            "a cut level or gradient is not finite, so the cutting-plane model has no floor \
             to hold its epigraph variable above",
        ));
    }

    start[k] = -highest - 1.0;

    let (value, gap, iters, status, _) = solve(&lp, &start, gaptol)?;
    let bound = -value - shift;

    // A certificate under `floor` is the one failure in the INSECURE direction.
    if bound < floor - FLOOR_TOL * floor.abs().max(1.0) {
        return Err(PyValueError::new_err(format!(
            "the certificate {bound} is below the model's own floor {floor} after {iters} \
             iterations with status \"{status}\": the facets admit no distribution or the path \
             broke down; check the facet rows are jointly satisfiable"
        )));
    }

    Ok((bound, gap, iters, status))
}

/// `polytope_cap` over the bare probability simplex.
pub(crate) fn simplex_cap(
    dim: usize,
    cuts: &[f64],
    levels: &[f64],
    gaptol: f64,
) -> PyResult<(f64, f64, usize, &'static str)> {
    let bare = Facets {
        eq: &[],
        eqb: &[],
        le: &[],
        leb: &[],
    };

    polytope_cap(dim, cuts, levels, &bare, gaptol)
}

/// `min c . v` subject to `A v = b`, `v >= 0`, `a` the `len(b) x len(c)` matrix row-major.
/// `start` is a dual point with `c - A^T start > 0` strictly. Returns `(value, gap, iters, y)`.
///
/// `value` is a rigorous LOWER bound on the minimum. `gap` is the last round's central-path gap
/// `n/t`: a diagnostic, not a bound.
#[pyfunction]
#[pyo3(signature = (c, a, b, start, gaptol=1e-9))]
pub(crate) fn lp_dual(
    c: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    start: Vec<f64>,
    gaptol: f64,
) -> PyResult<(f64, f64, usize, Vec<f64>)> {
    let n = c.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Err(PyValueError::new_err(
            "a linear programme needs at least one variable and one equality row",
        ));
    }

    if n > VAR_MAX || m > ROW_MAX {
        return Err(PyValueError::new_err(format!(
            "this solver is written for small dense programmes: got {n} variables against \
             a cap of {VAR_MAX} and {m} rows against a cap of {ROW_MAX}"
        )));
    }

    if a.len() != m * n {
        return Err(PyValueError::new_err(format!(
            "a must hold {} entries, {m} rows of {n} flattened row-major, got {}",
            m * n,
            a.len()
        )));
    }

    if start.len() != m {
        return Err(PyValueError::new_err(format!(
            "start is a dual point and carries one value per equality row: got {} for {m}",
            start.len()
        )));
    }

    check_pos("gaptol", gaptol)?;

    for x in c.iter().chain(a.iter()).chain(b.iter()) {
        if !x.is_finite() {
            return Err(PyValueError::new_err(
                "c, a and b must be finite: an infinite coefficient leaves no dual point to certify",
            ));
        }
    }

    let lp = Lp {
        n,
        m,
        c,
        a,
        b,
    };
    let (value, gap, iters, _, y) = solve(&lp, &start, gaptol)?;

    Ok((value, gap, iters, y))
}
