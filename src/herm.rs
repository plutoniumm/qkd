use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::check_pos;

// Complex Hermitian linear algebra and the log-barrier dual solver `dmcs::dm_secure` stands on.
// Kept apart from `sdp.rs` (real symmetric LMIs) on purpose; one register, no partial trace.
//
//   `Cone`: max gamma . y  s.t.  W_b - sum_i y_i Gamma_i^(b) >= 0 on every diagonal block b.
//
// The dual `gamma . y` at any strictly feasible `y` is a lower bound by weak duality: an early
// stop is looser, never wrong (Winick-Lutkenhaus-Coles step 2). The central-path primal
// `rho = S^-1 / t` (step 1's Frank-Wolfe) is feasible only to the centring, so its violation is
// returned and paid through WLC Theorem 3's epsilon'.

/// Six to ten sweeps converge at these sizes; the cap bounds a pathological input.
const SWEEP_MAX: usize = 60;

/// Off-diagonal Frobenius mass relative to the squared largest entry.
const OFF_TOL: f64 = 1e-30;

// Tuned apart from `sdp.rs`'s for one complex block of side up to 248: do not unify. `RETRY_MAX`
// is 6 in both and means `12^(1/64)` here, `15^(1/64)` there.

const NEWTON_MAX: usize = 80;

/// Squared Newton decrement.
const CENTRED: f64 = 1e-10;

const T_GROW: f64 = 12.0;

/// Below this the step is under the f64 resolution of `y`.
const STEP_MIN: f64 = 1e-13;

const ARMIJO: f64 = 0.25;

/// Each retry takes the square root of the growth.
const RETRY_MAX: usize = 6;

/// Relative to the Jacobi-scaled Hessian, whose diagonal is exactly 1.
const RIDGES: [f64; 4] = [0.0, 1e-13, 1e-11, 1e-9];

/// A dense square complex matrix, row-major, real and imaginary planes held apart.
pub(crate) struct Cx {
    pub(crate) n: usize,
    pub(crate) re: Vec<f64>,
    pub(crate) im: Vec<f64>,
}

impl Cx {
    pub(crate) fn zeros(n: usize) -> Cx {
        Cx {
            n,
            re: vec![0.0; n * n],
            im: vec![0.0; n * n],
        }
    }

    pub(crate) fn eye(n: usize) -> Cx {
        let mut a = Cx::zeros(n);

        for i in 0..n {
            a.re[i * n + i] = 1.0;
        }

        a
    }

    pub(crate) fn copy(&self) -> Cx {
        Cx {
            n: self.n,
            re: self.re.to_vec(),
            im: self.im.to_vec(),
        }
    }

    pub(crate) fn get(&self, i: usize, j: usize) -> (f64, f64) {
        let k = i * self.n + j;

        (self.re[k], self.im[k])
    }

    pub(crate) fn set(&mut self, i: usize, j: usize, re: f64, im: f64) {
        let k = i * self.n + j;
        self.re[k] = re;
        self.im[k] = im;
    }

    pub(crate) fn add(&mut self, i: usize, j: usize, re: f64, im: f64) {
        let k = i * self.n + j;
        self.re[k] += re;
        self.im[k] += im;
    }

    /// `self * other`.
    pub(crate) fn mul(&self, other: &Cx) -> Cx {
        let n = self.n;
        let mut out = Cx::zeros(n);

        for i in 0..n {
            for k in 0..n {
                let (ar, ai) = self.get(i, k);

                if ar == 0.0 && ai == 0.0 {
                    continue;
                }

                for j in 0..n {
                    let (br, bi) = other.get(k, j);
                    let idx = i * n + j;
                    out.re[idx] += ar * br - ai * bi;
                    out.im[idx] += ar * bi + ai * br;
                }
            }
        }

        out
    }

    /// Conjugate transpose.
    pub(crate) fn dag(&self) -> Cx {
        let n = self.n;
        let mut out = Cx::zeros(n);

        for i in 0..n {
            for j in 0..n {
                let (r, m) = self.get(j, i);
                out.set(i, j, r, -m);
            }
        }

        out
    }

    /// `(A + A^dag) / 2`: a product of Hermitian factors drifts off Hermitian in the last bits.
    pub(crate) fn symmetrise(&mut self) {
        let n = self.n;

        for i in 0..n {
            for j in i..n {
                let (ar, ai) = self.get(i, j);
                let (br, bi) = self.get(j, i);
                let re = 0.5 * (ar + br);
                let im = 0.5 * (ai - bi);
                self.set(i, j, re, im);
                self.set(j, i, re, -im);
            }
        }
    }

    pub(crate) fn trace(&self) -> f64 {
        let n = self.n;
        let mut s = 0.0;

        for i in 0..n {
            s += self.re[i * n + i];
        }

        s
    }

    pub(crate) fn peak(&self) -> f64 {
        let mut hi = 0.0f64;

        for k in 0..self.re.len() {
            hi = hi.max(self.re[k].abs()).max(self.im[k].abs());
        }

        hi
    }

    /// `L L^dag = self`; `None` is the certificate of non-definiteness.
    pub(crate) fn chol(&self) -> Option<Cx> {
        let n = self.n;
        let mut l = Cx::zeros(n);

        for i in 0..n {
            for j in 0..=i {
                let (mut sr, mut si) = self.get(i, j);

                for k in 0..j {
                    let (ar, ai) = l.get(i, k);
                    let (br, bi) = l.get(j, k);
                    sr -= ar * br + ai * bi;
                    si -= ai * br - ar * bi;
                }

                if i == j {
                    if !(sr > 0.0) {
                        return None;
                    }

                    l.set(i, i, sr.sqrt(), 0.0);
                } else {
                    let (d, _) = l.get(j, j);
                    l.set(i, j, sr / d, si / d);
                }
            }
        }

        Some(l)
    }
}

pub(crate) fn chol_logdet(l: &Cx) -> f64 {
    let n = l.n;
    let mut s = 0.0;

    for i in 0..n {
        s += l.re[i * n + i].ln();
    }

    2.0 * s
}

pub(crate) fn chol_inv(l: &Cx) -> Cx {
    let n = l.n;
    let mut inv = Cx::zeros(n);

    for i in 0..n {
        let (d, _) = l.get(i, i);
        inv.set(i, i, 1.0 / d, 0.0);

        for j in 0..i {
            let (mut sr, mut si) = (0.0f64, 0.0f64);

            for k in j..i {
                let (ar, ai) = l.get(i, k);
                let (br, bi) = inv.get(k, j);
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }

            let (dd, _) = l.get(i, i);
            inv.set(i, j, -sr / dd, -si / dd);
        }
    }

    let mut out = Cx::zeros(n);

    for i in 0..n {
        for j in 0..n {
            let (mut sr, mut si) = (0.0f64, 0.0f64);

            for k in i.max(j)..n {
                let (ar, ai) = inv.get(k, i);
                let (br, bi) = inv.get(k, j);
                sr += ar * br + ai * bi;
                si += ar * bi - ai * br;
            }

            out.set(i, j, sr, si);
        }
    }

    out
}

/// `Tr(AB)` for Hermitian `A`, `B`; the imaginary part cancels.
pub(crate) fn inner(a: &Cx, b: &Cx) -> f64 {
    let mut s = 0.0;

    for k in 0..a.re.len() {
        s += a.re[k] * b.re[k] + a.im[k] * b.im[k];
    }

    s
}

/// Cyclic Jacobi: `A = vecs diag(vals) vecs^dag`, eigenvectors in COLUMNS. Not QL: the matrix
/// logarithm needs small eigenvalues to high relative accuracy.
pub(crate) fn eig(a: &Cx) -> (Vec<f64>, Cx) {
    let n = a.n;
    let mut m = a.copy();
    let mut q = Cx::eye(n);
    let scale = m.peak();

    if scale == 0.0 {
        return (vec![0.0; n], q);
    }

    for _ in 0..SWEEP_MAX {
        let mut off = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let (r, s) = m.get(i, j);
                off += r * r + s * s;
            }
        }

        if off <= OFF_TOL * scale * scale {
            break;
        }

        for p in 0..n {
            for qq in (p + 1)..n {
                let (xr, xi) = m.get(p, qq);
                let mag = (xr * xr + xi * xi).sqrt();

                if mag == 0.0 {
                    continue;
                }

                // e^{-i phi} = conj(a_pq) / |a_pq|, so the block turns real symmetric.
                let (er, ei) = (xr / mag, -xi / mag);
                let app = m.re[p * n + p];
                let aqq = m.re[qq * n + qq];
                let theta = (aqq - app) / (2.0 * mag);
                let tan = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cos = 1.0 / (tan * tan + 1.0).sqrt();
                let sin = tan * cos;

                rotate(&mut m, p, qq, cos, sin, er, ei, true);
                rotate(&mut q, p, qq, cos, sin, er, ei, false);
            }
        }
    }

    let mut vals = vec![0.0f64; n];

    for i in 0..n {
        vals[i] = m.re[i * n + i];
    }

    (vals, q)
}

/// `A V`, or `V^dag A V` with `rows`; `V` is `[[c, s], [-s e^{-i phi}, c e^{-i phi}]]` on `p`, `q`.
fn rotate(m: &mut Cx, p: usize, q: usize, cos: f64, sin: f64, er: f64, ei: f64, rows: bool) {
    let n = m.n;

    for k in 0..n {
        let (pr, pi) = m.get(k, p);
        let (qr, qi) = m.get(k, q);

        // e^{-i phi} * A_kq
        let (tr, ti) = (qr * er - qi * ei, qr * ei + qi * er);
        m.set(k, p, cos * pr - sin * tr, cos * pi - sin * ti);
        m.set(k, q, sin * pr + cos * tr, sin * pi + cos * ti);
    }

    if !rows {
        return;
    }

    for k in 0..n {
        let (pr, pi) = m.get(p, k);
        let (qr, qi) = m.get(q, k);

        // e^{+i phi} * A_qk
        let (tr, ti) = (qr * er + qi * ei, qi * er - qr * ei);
        m.set(p, k, cos * pr - sin * tr, cos * pi - sin * ti);
        m.set(q, k, sin * pr + cos * tr, sin * pi + cos * ti);
    }
}

/// `V f(Lambda) V^dag` from an eigendecomposition and a scalar function.
pub(crate) fn funm(vals: &[f64], vecs: &Cx, f: impl Fn(f64) -> f64) -> Cx {
    let n = vecs.n;
    let mut scaled = Cx::zeros(n);

    for j in 0..n {
        let w = f(vals[j]);

        for i in 0..n {
            let (r, m) = vecs.get(i, j);
            scaled.set(i, j, r * w, m * w);
        }
    }

    let mut out = scaled.mul(&vecs.dag());
    out.symmetrise();

    out
}

/// Sparse Hermitian operator as a coordinate list; BOTH triangles listed.
pub(crate) struct Term {
    pub(crate) e: Vec<(usize, usize, f64, f64)>,
}

impl Term {
    /// `Re Tr(M G)` for Hermitian `M`.
    pub(crate) fn against(&self, m: &Cx) -> f64 {
        let mut s = 0.0;

        for &(i, j, gr, gi) in self.e.iter() {
            let (mr, mi) = m.get(j, i);
            s += mr * gr - mi * gi;
        }

        s
    }
}

/// `g.len() == Cone::gam.len()`; an empty term is a constraint the block does not see.
pub(crate) struct Block {
    pub(crate) w: Cx,
    pub(crate) g: Vec<Term>,
}

pub(crate) struct Cone {
    pub(crate) b: Vec<Block>,
    pub(crate) gam: Vec<f64>,
}

/// `y` is strictly feasible whatever `status` says; `rho` is `S_b^-1 / t` per block at the last
/// completed round.
pub(crate) struct Certificate {
    pub(crate) y: Vec<f64>,
    pub(crate) rho: Vec<Cx>,
    pub(crate) dual: f64,
    pub(crate) gap: f64,
    pub(crate) iters: usize,
    pub(crate) status: &'static str,
}

impl Cone {
    pub(crate) fn one(w: Cx, g: Vec<Term>, gam: Vec<f64>) -> Cone {
        Cone {
            b: vec![Block { w, g }],
            gam,
        }
    }

    /// Barrier parameter `nu`.
    pub(crate) fn size(&self) -> usize {
        self.b.iter().map(|blk| blk.w.n).sum()
    }

    pub(crate) fn slack(&self, y: &[f64]) -> Vec<Cx> {
        self.b
            .iter()
            .map(|blk| {
                let mut s = blk.w.copy();

                for (gi, &yi) in blk.g.iter().zip(y.iter()) {
                    if yi == 0.0 {
                        continue;
                    }

                    for &(i, j, gr, gm) in gi.e.iter() {
                        s.add(i, j, -yi * gr, -yi * gm);
                    }
                }

                s
            })
            .collect()
    }

    /// `-t (gamma . y) - sum_b log det S_b(y)`.
    fn barrier(&self, y: &[f64], t: f64) -> Option<f64> {
        let mut acc = 0.0;

        for (a, b) in self.gam.iter().zip(y.iter()) {
            acc -= t * a * b;
        }

        for s in self.slack(y).iter() {
            let l = s.chol()?;
            acc -= chol_logdet(&l);
        }

        if acc.is_finite() {
            Some(acc)
        } else {
            None
        }
    }

    /// Direction and squared decrement.
    fn newton(&self, y: &[f64], t: f64) -> Option<(Vec<f64>, f64)> {
        let n = self.gam.len();
        let mut grad = vec![0.0f64; n];
        let mut hess = vec![0.0f64; n * n];

        for a in 0..n {
            grad[a] = -t * self.gam[a];
        }

        for (blk, s) in self.b.iter().zip(self.slack(y).iter()) {
            let l = s.chol()?;
            let si = chol_inv(&l);

            for a in 0..n {
                grad[a] += blk.g[a].against(&si);
            }

            // H_ab = Tr(S^-1 Gamma_a S^-1 Gamma_b) on coordinate lists: never densify.
            for a in 0..n {
                for b in a..n {
                    let mut acc = 0.0;

                    for &(i, j, ar, ai) in blk.g[a].e.iter() {
                        for &(k, m, br, bi) in blk.g[b].e.iter() {
                            let (jr, ji) = si.get(j, k);
                            let (mr, mi) = si.get(m, i);
                            // (Gamma_a)_ij (S^-1)_jk (Gamma_b)_km (S^-1)_mi, real part.
                            let (pr, pi) = (ar * jr - ai * ji, ar * ji + ai * jr);
                            let (qr, qi) = (br * mr - bi * mi, br * mi + bi * mr);
                            acc += pr * qr - pi * qi;
                        }
                    }

                    hess[a * n + b] += acc;

                    if b > a {
                        hess[b * n + a] += acc;
                    }
                }
            }
        }

        // Marginal and quadrature terms differ by orders of magnitude in scale.
        let mut d = vec![0.0f64; n];

        for a in 0..n {
            let diag = hess[a * n + a];

            if !(diag > 0.0) {
                return None;
            }

            d[a] = 1.0 / diag.sqrt();
        }

        let mut scaled = vec![0.0f64; n * n];

        for a in 0..n {
            for b in 0..n {
                scaled[a * n + b] = hess[a * n + b] * d[a] * d[b];
            }
        }

        let mut rhs = vec![0.0f64; n];

        for a in 0..n {
            rhs[a] = -grad[a] * d[a];
        }

        for ridge in RIDGES.iter() {
            let mut trial = scaled.to_vec();

            for a in 0..n {
                trial[a * n + a] += ridge;
            }

            if let Some(f) = real_chol(&trial, n) {
                let w = real_solve(&f, n, &rhs);
                let mut dir = vec![0.0f64; n];
                let mut lam2 = 0.0;

                for a in 0..n {
                    dir[a] = w[a] * d[a];
                    lam2 -= grad[a] * dir[a];
                }

                if !dir.iter().all(|x| x.is_finite()) {
                    return None;
                }

                return Some((dir, lam2));
            }
        }

        None
    }

    /// Never leaves the strictly feasible region.
    fn advance(&self, y: &[f64], dir: &[f64], t: f64, lam2: f64) -> Option<Vec<f64>> {
        let n = self.gam.len();
        let base = self.barrier(y, t)?;
        let mut step = 1.0f64;

        while step > STEP_MIN {
            let mut trial = vec![0.0f64; n];

            for a in 0..n {
                trial[a] = y[a] + step * dir[a];
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

    /// `start` must be strictly feasible; stops at `nu/t <= gaptol`.
    pub(crate) fn run(&self, start: &[f64], gaptol: f64) -> Certificate {
        let nu = self.size() as f64;
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

            // A breakdown means `t` jumped too far, not that the path ended.
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

        let mut dual = 0.0;

        for (a, b) in self.gam.iter().zip(best.iter()) {
            dual += a * b;
        }

        let mut rho = Vec::with_capacity(self.b.len());
        let mut ok = true;

        for s in self.slack(&best).iter() {
            match s.chol() {
                None => {
                    ok = false;
                    rho.push(Cx::zeros(s.n));
                }
                Some(l) => {
                    let mut inv = chol_inv(&l);
                    inv.symmetrise();

                    for v in inv.re.iter_mut() {
                        *v /= held;
                    }

                    for v in inv.im.iter_mut() {
                        *v /= held;
                    }

                    rho.push(inv);
                }
            }
        }

        if !ok {
            status = "uncertified";
        }

        Certificate {
            y: best,
            rho,
            dual,
            gap: nu / held,
            iters,
            status,
        }
    }
}

// `real_chol`, `real_solve` duplicate `sdp::chol`, `sdp::chol_solve` on purpose: the solvers stay apart.

/// Real symmetric, row-major: the Newton Hessian is real whatever the blocks are.
fn real_chol(a: &[f64], n: usize) -> Option<Vec<f64>> {
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

fn real_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
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

pub(crate) fn read_herm(name: &str, re: &[f64], im: &[f64], tol: f64) -> PyResult<Cx> {
    if re.len() != im.len() {
        return Err(PyValueError::new_err(format!(
            "{name}: the real and imaginary planes must be the same length, got {} and {}",
            re.len(),
            im.len()
        )));
    }

    let n = (re.len() as f64).sqrt().round() as usize;

    if n * n != re.len() || n == 0 {
        return Err(PyValueError::new_err(format!(
            "{name}: {} entries is not a square matrix",
            re.len()
        )));
    }

    let a = Cx {
        n,
        re: re.to_vec(),
        im: im.to_vec(),
    };

    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = a.get(i, j);
            let (br, bi) = a.get(j, i);

            if (ar - br).abs() > tol || (ai + bi).abs() > tol {
                return Err(PyValueError::new_err(format!(
                    "{name} is not Hermitian: entry ({i}, {j}) reads {ar} + {ai}i against \
                     ({j}, {i})'s {br} + {bi}i, a mismatch above {tol}"
                )));
            }
        }
    }

    Ok(a)
}

/// `(vals, vec_re, vec_im)`, eigenvectors in COLUMNS; `vals` unsorted.
#[pyfunction]
#[pyo3(signature = (re, im))]
pub(crate) fn herm_eig<'py>(
    py: Python<'py>,
    re: Vec<f64>,
    im: Vec<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let a = read_herm("matrix", &re, &im, 1e-9)?;
    let (vals, vecs) = eig(&a);

    Ok((
        vals.into_pyarray(py),
        vecs.re.into_pyarray(py),
        vecs.im.into_pyarray(py),
    ))
}

/// `(max |A - V diag V^dag|, max |V^dag V - I|)`, absolute.
#[pyfunction]
#[pyo3(signature = (re, im))]
pub(crate) fn herm_residual(re: Vec<f64>, im: Vec<f64>) -> PyResult<(f64, f64)> {
    let a = read_herm("matrix", &re, &im, 1e-9)?;
    let n = a.n;
    let (vals, vecs) = eig(&a);
    let back = funm(&vals, &vecs, |x| x);
    let mut worst = 0.0f64;

    for k in 0..n * n {
        worst = worst.max((a.re[k] - back.re[k]).abs());
        worst = worst.max((a.im[k] - back.im[k]).abs());
    }

    let gram = vecs.dag().mul(&vecs);
    let mut off = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            let (r, m) = gram.get(i, j);
            let want = if i == j { 1.0 } else { 0.0 };
            off = off.max((r - want).abs()).max(m.abs());
        }
    }

    Ok((worst, off))
}

pub(crate) fn feasible(cone: &Cone, y: &[f64]) -> bool {
    cone.slack(y).iter().all(|s| s.chol().is_some())
}

/// `(dual, gap, iterations, status, y)`; `terms` is `(i, j, re, im)` per nonzero, both
/// triangles listed, `nnz` per constraint. `dual` is a proved lower bound whether or not the
/// path converged.
#[pyfunction]
#[pyo3(signature = (w_re, w_im, terms, nnz, gam, start, gaptol=1e-9))]
pub(crate) fn herm_cone(
    w_re: Vec<f64>,
    w_im: Vec<f64>,
    terms: Vec<f64>,
    nnz: Vec<usize>,
    gam: Vec<f64>,
    start: Vec<f64>,
    gaptol: f64,
) -> PyResult<(f64, f64, usize, String, Vec<f64>)> {
    check_pos("gaptol", gaptol)?;

    let w = read_herm("W", &w_re, &w_im, 1e-9)?;

    if gam.len() != nnz.len() || start.len() != nnz.len() {
        return Err(PyValueError::new_err(format!(
            "gam, nnz and start must all carry one value per constraint: got {}, {} and {}",
            gam.len(),
            nnz.len(),
            start.len()
        )));
    }

    let want: usize = nnz.iter().sum::<usize>() * 4;

    if terms.len() != want {
        return Err(PyValueError::new_err(format!(
            "terms must hold {want} numbers -- four per nonzero, (i, j, re, im) -- got {}",
            terms.len()
        )));
    }

    let mut g = Vec::with_capacity(nnz.len());
    let mut at = 0usize;

    for &count in nnz.iter() {
        let mut e = Vec::with_capacity(count);

        for _ in 0..count {
            let i = terms[at] as usize;
            let j = terms[at + 1] as usize;

            if i >= w.n || j >= w.n {
                return Err(PyValueError::new_err(format!(
                    "a constraint entry addresses ({i}, {j}) in a matrix of side {}",
                    w.n
                )));
            }

            e.push((i, j, terms[at + 2], terms[at + 3]));
            at += 4;
        }

        g.push(Term { e });
    }

    let cone = Cone::one(w, g, gam);

    if !feasible(&cone, &start) {
        return Err(PyValueError::new_err(
            "start is not strictly dual feasible: W - sum y_i Gamma_i is not positive \
             definite there, and the barrier is undefined",
        ));
    }

    let rep = cone.run(&start, gaptol);

    if !feasible(&cone, &rep.y) {
        return Err(PyValueError::new_err(
            "the returned point lost strict feasibility, so its objective certifies nothing",
        ));
    }

    if !rep.dual.is_finite() {
        return Err(PyValueError::new_err(
            "the dual objective is not finite at the returned point",
        ));
    }

    Ok((
        rep.dual,
        rep.gap,
        rep.iters,
        rep.status.to_string(),
        rep.y,
    ))
}

/// `herm_cone` over diagonal blocks of side `sizes`; `terms` is `(block, i, j, re, im)` per
/// nonzero. A block of side one is the sign condition `y_i >= 0`.
#[pyfunction]
#[pyo3(signature = (sizes, w_re, w_im, terms, nnz, gam, start, gaptol=1e-9))]
pub(crate) fn herm_split(
    sizes: Vec<usize>,
    w_re: Vec<f64>,
    w_im: Vec<f64>,
    terms: Vec<f64>,
    nnz: Vec<usize>,
    gam: Vec<f64>,
    start: Vec<f64>,
    gaptol: f64,
) -> PyResult<(f64, f64, usize, String, Vec<f64>)> {
    check_pos("gaptol", gaptol)?;

    if sizes.is_empty() || sizes.iter().any(|&n| n == 0) {
        return Err(PyValueError::new_err(
            "sizes must name at least one block and no block of side zero",
        ));
    }

    let want: usize = sizes.iter().map(|n| n * n).sum();

    if w_re.len() != want || w_im.len() != want {
        return Err(PyValueError::new_err(format!(
            "w_re and w_im must hold {want} numbers, the concatenated planes of blocks of \
             sides {sizes:?}, got {} and {}",
            w_re.len(),
            w_im.len()
        )));
    }

    if gam.len() != nnz.len() || start.len() != nnz.len() {
        return Err(PyValueError::new_err(format!(
            "gam, nnz and start must all carry one value per constraint: got {}, {} and {}",
            gam.len(),
            nnz.len(),
            start.len()
        )));
    }

    let nums: usize = nnz.iter().sum::<usize>() * 5;

    if terms.len() != nums {
        return Err(PyValueError::new_err(format!(
            "terms must hold {nums} numbers -- five per nonzero, (block, i, j, re, im) -- got {}",
            terms.len()
        )));
    }

    let mut blocks = Vec::with_capacity(sizes.len());
    let mut at = 0usize;

    for &n in sizes.iter() {
        let plane = n * n;
        let w = read_herm(
            "a block of W",
            &w_re[at..at + plane],
            &w_im[at..at + plane],
            1e-9,
        )?;
        blocks.push(Block {
            w,
            g: (0..nnz.len()).map(|_| Term { e: Vec::new() }).collect(),
        });
        at += plane;
    }

    let mut at = 0usize;

    for (c, &count) in nnz.iter().enumerate() {
        for _ in 0..count {
            let b = terms[at] as usize;
            let i = terms[at + 1] as usize;
            let j = terms[at + 2] as usize;

            if b >= blocks.len() {
                return Err(PyValueError::new_err(format!(
                    "a constraint entry names block {b} of {}",
                    blocks.len()
                )));
            }

            if i >= sizes[b] || j >= sizes[b] {
                return Err(PyValueError::new_err(format!(
                    "a constraint entry addresses ({i}, {j}) in block {b} of side {}",
                    sizes[b]
                )));
            }

            blocks[b].g[c].e.push((i, j, terms[at + 3], terms[at + 4]));
            at += 5;
        }
    }

    let cone = Cone { b: blocks, gam };

    if !feasible(&cone, &start) {
        return Err(PyValueError::new_err(
            "start is not strictly feasible: some block's W - sum y_i Gamma_i is not positive \
             definite there, and the barrier is undefined",
        ));
    }

    let rep = cone.run(&start, gaptol);

    if !feasible(&cone, &rep.y) {
        return Err(PyValueError::new_err(
            "the returned point lost strict feasibility, so its objective certifies nothing",
        ));
    }

    if !rep.dual.is_finite() {
        return Err(PyValueError::new_err(
            "the dual objective is not finite at the returned point",
        ));
    }

    Ok((rep.dual, rep.gap, rep.iters, rep.status.to_string(), rep.y))
}
