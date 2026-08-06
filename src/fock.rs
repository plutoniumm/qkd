use numpy::{IntoPyArray, PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::std::{check_finite, g_ent, ln_fact};

// Truncated Fock states on levels 0..d-1. Conventions as gaussian.rs: hbar = 1,
// x = (a + a^dagger)/sqrt(2), p = (a - a^dagger)/(i sqrt(2)), vacuum variance
// 1/2; a coherent state of amplitude beta sits at x = sqrt(2) Re(beta). Grids
// are normalised in dx dp, not d^2 alpha: W_vac(0, 0) = 1/pi, Q_vac = 1/(2 pi).
// A truncated matrix is renormalised; `discarded()` and `tail()` report that.
// Wigner kernel (Cahill & Glauber), sqrt(n!/(n+k)!) and (2 alpha*)^k regrouped
// to stay in range:  Delta_{n, n+k} = (1/pi) (-1)^n (2 alpha*)^k
// sqrt(n!/(n+k)!) e^{-r^2} L_n^(k)(2 r^2),  r^2 = x^2 + p^2.

/// Level cap: past it the Laguerre scale eats f64's exponent range. `pnr.rs` imports it, the
/// two layers indexing the same truncated space.
pub(crate) const MAX_D: usize = 512;

/// Levels at the top of the ladder that `tail()` counts as "the edge".
const EDGE: usize = 3;

/// Squared radius past which every quasi-probability is zero to f64.
const FAR: f64 = 64.0 * MAX_D as f64;

const SQRT2: f64 = std::f64::consts::SQRT_2;

const PI: f64 = std::f64::consts::PI;

/// Eigenvalues below this are dropped from an entropy sum as round-off zeros.
const TINY: f64 = 1e-14;

fn check_cutoff(d: usize) -> PyResult<()> {
    if (1..=MAX_D).contains(&d) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "cutoff must be in 1..={MAX_D}, got {d}"
        )))
    }
}

/// Drift from the mean step still read as uniform; round-off is parts in 1e13.
const GRID_SLACK: f64 = 1e-6;

/// Refuses a non-uniform axis: only there is the step an integration measure.
fn grid_step(name: &str, v: &[f64]) -> PyResult<f64> {
    if v.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "{name} needs at least two points, got {}",
            v.len()
        )));
    }
    for x in v {
        check_finite(name, *x)?;
    }

    let step = (v[v.len() - 1] - v[0]) / ((v.len() - 1) as f64);
    if !(step > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must increase, got {} .. {}",
            v[0],
            v[v.len() - 1]
        )));
    }

    let slack = GRID_SLACK * step;
    for i in 0..(v.len() - 1) {
        let gap = v[i + 1] - v[i];
        if (gap - step).abs() > slack {
            return Err(PyValueError::new_err(format!(
                "{name} must be evenly spaced: its mean step is {step:.6e} but \
                 points {i} and {} are {gap:.6e} apart. A quadrature over a \
                 non-uniform axis is not an integral",
                i + 1
            )));
        }
    }

    Ok(step)
}

/// Laguerre recursions in flight. Do not raise it past 6: at 8, chains x 5 live
/// values cross aarch64's 32 FP registers and the recursion spills.
const CHAINS: usize = 6;

// Tabulating the radicals is a 1.6x pessimisation; a * (1/b) costs 10 ulp in W.

/// The (n, k)-only parts of a step, so chains at that (n, k) share the sqrts.
fn lag_coef(n: usize, k: usize) -> (f64, f64, f64) {
    let (nf, kf) = (n as f64, k as f64);

    (
        2.0 * nf + 1.0 + kf,
        (nf * (nf + kf)).sqrt(),
        ((nf + 1.0) * (nf + 1.0 + kf)).sqrt(),
    )
}

/// One step of the scaled recursion d_{n,k}(z) = L_n^(k)(z)/sqrt(binom(n+k, n)),
/// whose scaling keeps every intermediate in f64's range. d_{0,k} = 1.
fn lag_step(c: (f64, f64, f64), z: f64, dn: f64, dm: f64) -> f64 {
    ((c.0 - z) * dn - c.1 * dm) / c.2
}

/// `lag_step` at one (n, k); never respell the arithmetic.
fn lag_next(n: usize, k: usize, z: f64, dn: f64, dm: f64) -> f64 {
    lag_step(lag_coef(n, k), z, dn, dm)
}

/// One past the last non-zero row of each superdiagonal, 0 for one zero
/// throughout. Parity-restricted states are half zeros; skipping those is exact.
fn live_rows(d: usize, re: &[f64], im: &[f64]) -> Vec<usize> {
    let mut out = vec![0_usize; d];
    for (k, slot) in out.iter_mut().enumerate() {
        for n in 0..(d - k) {
            let idx = (n + k) * d + n;
            if re[idx] != 0.0 || im[idx] != 0.0 {
                *slot = n + 1;
            }
        }
    }

    out
}

/// Whether an axis is its own bit-exact mirror: true because `window()` mirrors
/// a half, false for a plain `np.linspace(-a, a, n)` off a dyadic step.
fn mirrored(v: &[f64]) -> bool {
    let n = v.len();

    (0..n).all(|i| v[i] == -v[n - 1 - i])
}

/// Same numbers on both axes, admitting the x <-> p swap: `p*p + x*x` is exact.
fn identical(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// The symmetries an (xs, ps) grid has. The O(d^2) half of the kernel sees the
/// point only through r^2 = x^2 + p^2, so one class covers up to 8 points.
struct Fold {
    nx: usize,
    np: usize,
    xsym: bool,
    psym: bool,
    same: bool,
}

impl Fold {
    fn of(xs: &[f64], ps: &[f64]) -> Self {
        Self {
            nx: xs.len(),
            np: ps.len(),
            xsym: mirrored(xs),
            psym: mirrored(ps),
            same: identical(xs, ps),
        }
    }

    /// One representative per r^2 class, as (row, column) index pairs.
    fn cells(&self) -> Vec<(usize, usize)> {
        let hx = if self.xsym { self.nx.div_ceil(2) } else { self.nx };
        let hp = if self.psym { self.np.div_ceil(2) } else { self.np };
        let mut out = Vec::with_capacity(hp * hx);
        for a in 0..hp {
            let lo = if self.same { a } else { 0 };
            for b in lo..hx {
                out.push((a, b));
            }
        }

        out
    }

    /// Cells per r^2 class; an upper bound, as an on-axis orbit repeats cells with the same bits.
    fn reach(&self) -> usize {
        let quad = (1 + usize::from(self.psym)) * (1 + usize::from(self.xsym));

        if self.same {
            2 * quad
        } else {
            quad
        }
    }

    /// The cells sharing (a, b)'s radius, the `reach()` distinct ones first.
    fn orbit(&self, a: usize, b: usize) -> [(usize, usize); 8] {
        let rows = [a, self.np - 1 - a];
        let cols = [b, self.nx - 1 - b];
        let mut out = [(a, b); 8];
        let mut n = 0;
        for i in 0..(1 + usize::from(self.psym)) {
            for j in 0..(1 + usize::from(self.xsym)) {
                out[n] = (rows[i], cols[j]);
                n += 1;
            }
        }
        if self.same {
            let turn = [b, self.np - 1 - b];
            let back = [a, self.nx - 1 - a];
            for i in 0..(1 + usize::from(self.psym)) {
                for j in 0..(1 + usize::from(self.xsym)) {
                    out[n] = (turn[i], back[j]);
                    n += 1;
                }
            }
        }

        out
    }
}

/// Ascending, cyclic Jacobi; round-off floors the off-diagonal sum at ~d^2 eps^2 ||A||_F^2.
fn eigvals(d: usize, re: &[f64], im: &[f64]) -> Vec<f64> {
    let mut ar = re.to_vec();
    let mut ai = im.to_vec();
    let fro: f64 = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| r * r + i * i)
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let eps2 = f64::EPSILON * f64::EPSILON;
    let settled = f64::max(1e-30, 4.0 * (d * d) as f64 * eps2 * fro);
    for _ in 0..64 {
        let mut off = 0.0;
        for i in 0..d {
            for j in (i + 1)..d {
                off += ar[i * d + j] * ar[i * d + j] + ai[i * d + j] * ai[i * d + j];
            }
        }
        if off <= settled {
            break;
        }

        for p in 0..d {
            for q in (p + 1)..d {
                let mag = ar[p * d + q].hypot(ai[p * d + q]);
                // Divide-by-zero guard, not a convergence threshold: the phase
                // is ar/mag, parity zeros abound, 1e-300 dodges the subnormals.
                if mag <= 1e-300 {
                    continue;
                }

                // Phase column q (and row q, conjugately) so the pivot is +mag.
                let (er, ei) = (ar[p * d + q] / mag, -ai[p * d + q] / mag);
                for i in 0..d {
                    if i == q {
                        continue;
                    }
                    let (xr, xi) = (ar[i * d + q], ai[i * d + q]);
                    ar[i * d + q] = xr * er - xi * ei;
                    ai[i * d + q] = xr * ei + xi * er;
                    ar[q * d + i] = ar[i * d + q];
                    ai[q * d + i] = -ai[i * d + q];
                }

                let (app, aqq) = (ar[p * d + p], ar[q * d + q]);
                let th = (aqq - app) / (2.0 * mag);
                let root = (th * th + 1.0).sqrt();
                let t = if th >= 0.0 {
                    1.0 / (th + root)
                } else {
                    -1.0 / (root - th)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                ar[p * d + p] = app - t * mag;
                ar[q * d + q] = aqq + t * mag;
                ar[p * d + q] = 0.0;
                ai[p * d + q] = 0.0;
                ar[q * d + p] = 0.0;
                ai[q * d + p] = 0.0;
                for i in 0..d {
                    if i == p || i == q {
                        continue;
                    }
                    let (pr, pi) = (ar[i * d + p], ai[i * d + p]);
                    let (qr, qi) = (ar[i * d + q], ai[i * d + q]);
                    ar[i * d + p] = c * pr - s * qr;
                    ai[i * d + p] = c * pi - s * qi;
                    ar[i * d + q] = s * pr + c * qr;
                    ai[i * d + q] = s * pi + c * qi;
                    ar[p * d + i] = ar[i * d + p];
                    ai[p * d + i] = -ai[i * d + p];
                    ar[q * d + i] = ar[i * d + q];
                    ai[q * d + i] = -ai[i * d + q];
                }
            }
        }
    }

    let mut out: Vec<f64> = (0..d).map(|i| ar[i * d + i]).collect();
    out.sort_by(f64::total_cmp);

    out
}

/// A single-mode state in the truncated number basis, plus the discarded norm.
/// Units are gaussian.rs's (hbar = 1, vacuum variance 1/2), not shot-noise ones.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct FockState {
    d: usize,
    re: Vec<f64>,
    im: Vec<f64>,
    live: Vec<usize>,
    lost: f64,
}

impl FockState {
    /// The only assembly point, so the live-row scan cannot be skipped.
    fn build(d: usize, re: Vec<f64>, im: Vec<f64>, lost: f64) -> Self {
        let live = live_rows(d, &re, &im);

        Self {
            d,
            re,
            im,
            live,
            lost: lost.clamp(0.0, 1.0),
        }
    }

    /// Outer product of a ket with itself, normalising on the way in.
    fn from_amps(mut ar: Vec<f64>, mut ai: Vec<f64>, lost: f64) -> PyResult<Self> {
        let d = ar.len();
        let norm: f64 = (0..d).map(|n| ar[n] * ar[n] + ai[n] * ai[n]).sum();
        if !(norm > 0.0) {
            return Err(PyValueError::new_err(
                "the state has no weight inside the cutoff: raise it",
            ));
        }

        let scale = 1.0 / norm.sqrt();
        for n in 0..d {
            ar[n] *= scale;
            ai[n] *= scale;
        }

        let mut re = vec![0.0; d * d];
        let mut im = vec![0.0; d * d];
        for m in 0..d {
            for n in 0..d {
                re[m * d + n] = ar[m] * ar[n] + ai[m] * ai[n];
                im[m * d + n] = ai[m] * ar[n] - ar[m] * ai[n];
            }
        }

        Ok(Self::build(d, re, im, lost))
    }

    /// Coherent amplitudes <n|beta> on 0..d-1, unnormalised, with the norm kept.
    fn coh_amps(beta: (f64, f64), d: usize) -> (Vec<f64>, Vec<f64>, f64) {
        let nb = beta.0 * beta.0 + beta.1 * beta.1;
        let mut ar = vec![0.0; d];
        let mut ai = vec![0.0; d];
        ar[0] = (-0.5 * nb).exp();
        for n in 1..d {
            let inv = 1.0 / (n as f64).sqrt();
            ar[n] = (ar[n - 1] * beta.0 - ai[n - 1] * beta.1) * inv;
            ai[n] = (ar[n - 1] * beta.1 + ai[n - 1] * beta.0) * inv;
        }
        let kept: f64 = (0..d).map(|n| ar[n] * ar[n] + ai[n] * ai[n]).sum();

        (ar, ai, kept)
    }

    /// Angle-free half of the Wigner sum, N radii at a time; `ok` false where W and Q are zero.
    /// Chains, not SIMD: latency-bound on `cur`.
    ///
    /// ANY PORT MUST KEEP THE `big`/`ls` RENORMALISATION. QuTiP `_wigner_clenshaw` and
    /// QuantumOptics.jl `src/phasespace.jl` shed no exponent: a verbatim port NaNs at the peak
    /// while `discarded` and `physical` read healthy. Remedy: doi:10.1007/s10915-024-02725-9.
    fn cols_n<const N: usize>(
        &self,
        rad: &[f64; N],
        ok: &mut [bool; N],
        pref: &mut [f64],
        sr: &mut [f64],
        si: &mut [f64],
    ) {
        let d = self.d;
        let mut z = [0.0; N];
        let mut lt = [0.0; N];
        let mut lr = [0.0; N];
        let mut big = [0.0; N];
        let mut shed = [0.0; N];
        for j in 0..N {
            // Past FAR the envelope wins by e^{-27000}; NaN lands here too. A dead chain walks r^2 = 0.
            ok[j] = !(rad[j] > FAR) && !rad[j].is_nan();
            let r2 = if ok[j] { rad[j] } else { 0.0 };

            // |T_k| = (sqrt(2) r)^k e^{-r^2} / sqrt(k!) in logs; arg(T_k) is `wigner_tail`'s.
            z[j] = 2.0 * r2;
            lt[j] = -r2;
            lr[j] = if r2 > 0.0 {
                0.5 * z[j].ln()
            } else {
                f64::NEG_INFINITY
            };

            // One step multiplies by at most z + 2d + 1: 53 decades under f64::MAX, 58 over MIN_POSITIVE.
            big[j] = (f64::MAX / (4.0 * (z[j] + 2.0 * d as f64 + 2.0))).min(1e250);
            shed[j] = big[j].ln();
        }

        for k in 0..d {
            if k > 0 {
                let dk = 0.5 * (k as f64).ln();
                for j in 0..N {
                    lt[j] += lr[j] - dk;
                }
            }

            // An exactly-zero superdiagonal contributes +-0.0; see `live_rows`.
            let rows = self.live[k];
            if rows == 0 {
                for j in 0..N {
                    pref[j * d + k] = 0.0;
                    sr[j * d + k] = 0.0;
                    si[j * d + k] = 0.0;
                }

                continue;
            }

            let mut ar = [0.0; N];
            let mut ai = [0.0; N];
            let mut ls = [0.0; N];
            let mut prev = [0.0; N];
            let mut cur = [1.0; N];
            for n in 0..rows {
                let idx = (n + k) * d + n;
                let flip = n % 2 == 1;
                let rr = if flip { -self.re[idx] } else { self.re[idx] };
                let ii = if flip { -self.im[idx] } else { self.im[idx] };
                let c = lag_coef(n, k);
                for j in 0..N {
                    ar[j] += rr * cur[j];
                    ai[j] += ii * cur[j];
                    let next = lag_step(c, z[j], cur[j], prev[j]);
                    prev[j] = cur[j];
                    cur[j] = next;
                    if cur[j].abs() > big[j] {
                        cur[j] /= big[j];
                        prev[j] /= big[j];
                        ar[j] /= big[j];
                        ai[j] /= big[j];
                        ls[j] += shed[j];
                    }
                }
            }

            for j in 0..N {
                pref[j * d + k] = (lt[j] + ls[j]).exp();
                sr[j * d + k] = ar[j];
                si[j * d + k] = ai[j];
            }
        }
    }

    /// The angle-dependent O(d) remainder. Written once for both drivers:
    /// `cs*sr - sn*si` spelt twice can contract into an FMA in one place only.
    fn wigner_tail(&self, x: f64, p: f64, pref: &[f64], sr: &[f64], si: &[f64]) -> f64 {
        let phi = (-p).atan2(x);
        let mut acc = 0.0;
        for k in 0..self.d {
            if pref[k] == 0.0 {
                continue;
            }

            let (sn, cs) = ((k as f64) * phi).sin_cos();
            let term = pref[k] * (cs * sr[k] - sn * si[k]);
            acc += if k == 0 { term } else { 2.0 * term };
        }

        acc / PI
    }

    /// Husimi Q at one phase-space point, in the dx dp normalisation. `coh_amps`'s recursion at
    /// a different rounding order -- `-0.25 (x^2 + p^2)` here against `-0.5 |beta|^2` there --
    /// and `assertHusimi`'s tolerance is sized to that round-off.
    fn husimi_at(&self, x: f64, p: f64, cr: &mut [f64], ci: &mut [f64]) -> f64 {
        let d = self.d;
        let (alr, ali) = (x / SQRT2, p / SQRT2);
        cr[0] = (-0.25 * (x * x + p * p)).exp();
        ci[0] = 0.0;
        for n in 1..d {
            let inv = 1.0 / (n as f64).sqrt();
            let nr = (cr[n - 1] * alr - ci[n - 1] * ali) * inv;
            ci[n] = (cr[n - 1] * ali + ci[n - 1] * alr) * inv;
            cr[n] = nr;
        }

        // <alpha|rho|alpha> = sum_{mn} conj(c_m) rho_{mn} c_n, real by symmetry.
        let mut acc = 0.0;
        for m in 0..d {
            let mut ur = 0.0;
            let mut ui = 0.0;
            for n in 0..d {
                let (rr, ri) = (self.re[m * d + n], self.im[m * d + n]);
                ur += rr * cr[n] - ri * ci[n];
                ui += rr * ci[n] + ri * cr[n];
            }
            acc += cr[m] * ur + ci[m] * ui;
        }

        acc / (2.0 * PI)
    }

    /// Row-major in (len(ps), len(xs)); same bits whatever the grid's symmetry.
    fn wigner_grid(&self, xs: &[f64], ps: &[f64]) -> Vec<f64> {
        let nx = xs.len();
        let mut out = vec![0.0; ps.len() * nx];
        let fold = Fold::of(xs, ps);
        let reach = fold.reach();
        let cells = fold.cells();
        let vals = match reach {
            8 => self.fold_vals::<CHAINS, 8>(xs, ps, &fold, &cells),
            4 => self.fold_vals::<CHAINS, 4>(xs, ps, &fold, &cells),
            2 => self.fold_vals::<CHAINS, 2>(xs, ps, &fold, &cells),
            _ => self.fold_vals::<CHAINS, 1>(xs, ps, &fold, &cells),
        };
        for (&(a, b), got) in cells.iter().zip(vals.chunks(reach)) {
            for (slot, (i, j)) in got.iter().zip(fold.orbit(a, b)) {
                out[i * nx + j] = *slot;
            }
        }

        out
    }

    /// N radii per group, R points per cell; a short final group pads r^2 = 0.
    fn fold_vals<const N: usize, const R: usize>(
        &self,
        xs: &[f64],
        ps: &[f64],
        fold: &Fold,
        cells: &[(usize, usize)],
    ) -> Vec<f64> {
        let d = self.d;
        let mut vals = vec![0.0; cells.len() * R];
        // Rayon's own splitter sizes the task; a forced grain starves the pool.
        vals.par_chunks_mut(N * R)
            .zip(cells.par_chunks(N))
            .for_each_init(
                || (vec![0.0; N * d], vec![0.0; N * d], vec![0.0; N * d]),
                |scratch, (slots, group)| {
                    let (pref, sr, si) = scratch;
                    let mut rad = [0.0; N];
                    let mut ok = [false; N];
                    for (slot, &(a, b)) in rad.iter_mut().zip(group.iter()) {
                        *slot = xs[b] * xs[b] + ps[a] * ps[a];
                    }
                    self.cols_n::<N>(&rad, &mut ok, pref, sr, si);
                    for (j, (&(a, b), got)) in
                        group.iter().zip(slots.chunks_exact_mut(R)).enumerate()
                    {
                        if !ok[j] {
                            continue;
                        }

                        let lo = j * d;
                        let (pf, cr, ci) = (&pref[lo..], &sr[lo..], &si[lo..]);
                        let orb = fold.orbit(a, b);
                        for t in 0..R {
                            let (i, jj) = orb[t];
                            got[t] = self.wigner_tail(xs[jj], ps[i], pf, cr, ci);
                        }
                    }
                },
            );

        vals
    }

    fn raw_moments(&self) -> (Vec<f64>, Vec<f64>) {
        let d = self.d;
        let (mut a1r, mut a1i) = (0.0, 0.0);
        let (mut a2r, mut a2i) = (0.0, 0.0);
        let mut nbar = 0.0;
        for n in 0..d {
            nbar += (n as f64) * self.re[n * d + n];
            if n + 1 < d {
                let w = ((n + 1) as f64).sqrt();
                a1r += w * self.re[(n + 1) * d + n];
                a1i += w * self.im[(n + 1) * d + n];
            }
            if n + 2 < d {
                let w = (((n + 2) * (n + 1)) as f64).sqrt();
                a2r += w * self.re[(n + 2) * d + n];
                a2i += w * self.im[(n + 2) * d + n];
            }
        }

        let (mx, mp) = (SQRT2 * a1r, SQRT2 * a1i);
        let base = nbar + 0.5;
        let cov = vec![
            base + a2r - mx * mx,
            a2i - mx * mp,
            a2i - mx * mp,
            base - a2r - mp * mp,
        ];

        (vec![mx, mp], cov)
    }
}

#[pymethods]
impl FockState {
    /// The number state |n> inside `cutoff` levels. No truncation error.
    #[staticmethod]
    fn fock(n: usize, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        if n >= cutoff {
            return Err(PyValueError::new_err(format!(
                "|{n}> does not fit under a cutoff of {cutoff} levels"
            )));
        }

        let mut ar = vec![0.0; cutoff];
        ar[n] = 1.0;

        Self::from_amps(ar, vec![0.0; cutoff], 0.0)
    }

    /// The coherent state at (x, p), beta = (x + i p)/sqrt(2), matching the
    /// Gaussian layer's `Coherent`. Poissonian tails make `discarded()` non-zero.
    #[staticmethod]
    fn coherent(x: f64, p: f64, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        check_finite("x", x)?;
        check_finite("p", p)?;

        let beta = (x / SQRT2, p / SQRT2);
        let (ar, ai, kept) = Self::coh_amps(beta, cutoff);

        Self::from_amps(ar, ai, 1.0 - kept)
    }

    /// The even (`odd = false`) or odd normalised superposition of the coherent
    /// states at +-(x, p): parity-restricted, so an odd cat has parity exactly -1.
    #[staticmethod]
    fn cat(x: f64, p: f64, odd: bool, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        check_finite("x", x)?;
        check_finite("p", p)?;

        let beta = (x / SQRT2, p / SQRT2);
        let nb = beta.0 * beta.0 + beta.1 * beta.1;
        let (mut ar, mut ai, _) = Self::coh_amps(beta, cutoff);
        let want = usize::from(odd);
        let mut kept = 0.0;
        for n in 0..cutoff {
            if n % 2 == want {
                ar[n] *= 2.0;
                ai[n] *= 2.0;
                kept += ar[n] * ar[n] + ai[n] * ai[n];
            } else {
                ar[n] = 0.0;
                ai[n] = 0.0;
            }
        }

        // Exact norm of |beta> +- |-beta> before truncation: 2(1 -+ e^{-2|b|^2}).
        let full = if odd {
            2.0 * (1.0 - (-2.0 * nb).exp())
        } else {
            2.0 * (1.0 + (-2.0 * nb).exp())
        };
        if !(full > 0.0) {
            return Err(PyValueError::new_err(
                "an odd cat needs a non-zero amplitude: |beta> - |-beta> vanishes at beta = 0",
            ));
        }

        Self::from_amps(ar, ai, 1.0 - kept / full)
    }

    /// The squeezed vacuum S(r)|0>, x squeezed for r > 0, matching the Gaussian
    /// layer's `Squeezed(r)` (symplectic map diag(e^-r, e^r)). Even levels only.
    #[staticmethod]
    fn squeezed(r: f64, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        check_finite("r", r)?;

        let th = r.tanh();
        let mut ar = vec![0.0; cutoff];
        ar[0] = 1.0 / r.cosh().sqrt();
        let mut m = 1;
        while 2 * m < cutoff {
            let mf = m as f64;
            ar[2 * m] = ar[2 * m - 2] * (-th) * ((2.0 * mf - 1.0) / (2.0 * mf)).sqrt();
            m += 1;
        }
        let kept: f64 = ar.iter().map(|c| c * c).sum();

        Self::from_amps(ar, vec![0.0; cutoff], 1.0 - kept)
    }

    /// Thermal state of mean photon number nbar: diagonal, nbar^n/(1 + nbar)^{n+1}.
    #[staticmethod]
    fn thermal(nbar: f64, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        if !(nbar.is_finite() && nbar >= 0.0) {
            return Err(PyValueError::new_err(format!("nbar must be >= 0, got {nbar}")));
        }

        let ratio = nbar / (1.0 + nbar);
        let mut pops = vec![0.0; cutoff];
        pops[0] = 1.0 / (1.0 + nbar);
        for n in 1..cutoff {
            pops[n] = pops[n - 1] * ratio;
        }
        let kept: f64 = pops.iter().sum();
        if !(kept > 0.0) {
            return Err(PyValueError::new_err(
                "the thermal state has no weight inside the cutoff: raise it",
            ));
        }

        let mut re = vec![0.0; cutoff * cutoff];
        for n in 0..cutoff {
            re[n * cutoff + n] = pops[n] / kept;
        }

        Ok(Self::build(cutoff, re, vec![0.0; cutoff * cutoff], 1.0 - kept))
    }

    /// Finite-energy GKP, `logical` = 0 or 1, in the symmetric finite-squeezing
    /// approximation (peak variance delta^2/2, envelope 1/(2 delta^2), so
    /// O(delta^2) from e^{-eps n}|GKP_ideal>); delta = 1 is the vacuum width,
    /// squeezing -20 log10(delta) dB, x_s = 2 s sqrt(pi), +sqrt(pi) for |1_L>.
    #[staticmethod]
    fn gkp(logical: usize, delta: f64, cutoff: usize) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        if logical > 1 {
            return Err(PyValueError::new_err(format!(
                "logical must be 0 or 1, got {logical}"
            )));
        }
        // The floor is where the state stops fitting MAX_D: nbar = 155 for
        // gkp(0, 0.05, 512). The ceiling is sanity; negativity is gone by 1.3.
        if !(delta.is_finite() && (0.05..=2.0).contains(&delta)) {
            return Err(PyValueError::new_err(format!(
                "delta must be in [0.05, 2], got {delta}"
            )));
        }

        let root = PI.sqrt();
        let shift = if logical == 0 { 0.0 } else { root };
        // Covers the envelope and the top Hermite turning point sqrt(2d - 1).
        let span = f64::max(6.0 / delta, (2.0 * cutoff as f64).sqrt() + 5.0);
        // ~6x inside both Nyquist bounds: pi delta/6 for the peaks, pi/32 for the top Hermite.
        let step = f64::min(delta / 12.0, 0.02);
        let half = (span / step).ceil() as usize;
        let peaks = ((span + 6.0 * delta) / (2.0 * root)).ceil() as i64 + 1;

        let comb: Vec<(f64, f64)> = (-peaks..=peaks)
            .map(|s| {
                let xs = 2.0 * (s as f64) * root + shift;

                (xs, (-0.5 * delta * delta * xs * xs).exp())
            })
            .collect();

        let mut acc = vec![0.0; cutoff];
        let mut norm = 0.0;
        for i in 0..(2 * half + 1) {
            let x = -span + (i as f64) * step;
            let mut psi = 0.0;
            for (xs, env) in comb.iter().copied() {
                psi += env * (-(x - xs) * (x - xs) / (2.0 * delta * delta)).exp();
            }
            norm += psi * psi * step;

            let mut h0 = PI.powf(-0.25) * (-0.5 * x * x).exp();
            let mut h1 = SQRT2 * x * h0;
            acc[0] += psi * h0 * step;
            if cutoff > 1 {
                acc[1] += psi * h1 * step;
            }
            for n in 1..cutoff.saturating_sub(1) {
                let nf = n as f64;
                let h2 = x * (2.0 / (nf + 1.0)).sqrt() * h1 - (nf / (nf + 1.0)).sqrt() * h0;
                acc[n + 1] += psi * h2 * step;
                h0 = h1;
                h1 = h2;
            }
        }
        if !(norm > 0.0) {
            return Err(PyValueError::new_err("the GKP wavefunction integrated to zero"));
        }

        let scale = 1.0 / norm.sqrt();
        for c in acc.iter_mut() {
            *c *= scale;
        }
        let kept: f64 = acc.iter().map(|c| c * c).sum();

        Self::from_amps(acc, vec![0.0; cutoff], 1.0 - kept)
    }

    /// A state given as an explicit d x d density matrix, row-major re and im.
    /// Validated for Hermiticity, unit trace and positive semidefiniteness (the
    /// Fock bona fide condition). `discarded()` is 0 here; `tail()` still applies.
    #[staticmethod]
    fn from_matrix(re: Vec<f64>, im: Vec<f64>) -> PyResult<Self> {
        if re.len() != im.len() {
            return Err(PyValueError::new_err(format!(
                "real and imaginary parts must match in length, got {} and {}",
                re.len(),
                im.len()
            )));
        }

        let d = (re.len() as f64).sqrt().round() as usize;
        if d * d != re.len() || d == 0 {
            return Err(PyValueError::new_err(format!(
                "the matrix must be square and non-empty, got {} entries",
                re.len()
            )));
        }
        check_cutoff(d)?;
        for (i, x) in re.iter().chain(im.iter()).enumerate() {
            if !x.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "matrix entries must be finite, entry {i} is {x}"
                )));
            }
        }

        // One absolute tolerance for all three because unit trace fixes the scale:
        // |rho_mn| <= 1/2, every eigenvalue in [0, 1], unlike gaussian.rs's V.
        for i in 0..d {
            for j in 0..d {
                let gap = (re[i * d + j] - re[j * d + i]).abs() + (im[i * d + j] + im[j * d + i]).abs();
                if gap > 1e-9 {
                    return Err(PyValueError::new_err(format!(
                        "rho must be Hermitian, entries ({i}, {j}) and ({j}, {i}) differ by {gap:.3e}"
                    )));
                }
            }
        }

        let trace: f64 = (0..d).map(|i| re[i * d + i]).sum();
        if (trace - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "rho must have unit trace, got {trace:.9}"
            )));
        }

        let low = eigvals(d, &re, &im).first().copied().unwrap_or(0.0);
        if low < -1e-9 {
            return Err(PyValueError::new_err(format!(
                "rho must be positive semidefinite, its lowest eigenvalue is {low:.3e}: \
                 Hermitian with unit trace is not enough to be a quantum state"
            )));
        }

        Ok(Self::build(d, re, im, 0.0))
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.d
    }

    /// Fraction of the norm this state's construction discarded.
    fn discarded(&self) -> f64 {
        self.lost
    }

    /// Density matrix as flat row-major (real, imaginary) float64 arrays; copies.
    fn matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        (self.re.to_pyarray(py), self.im.to_pyarray(py))
    }

    /// Photon-number distribution P(n) = rho_{nn}.
    fn populations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p: Vec<f64> = (0..self.d).map(|n| self.re[n * self.d + n]).collect();

        p.into_pyarray(py)
    }

    /// Mean photon number <a^dagger a>.
    fn photons(&self) -> f64 {
        (0..self.d)
            .map(|n| (n as f64) * self.re[n * self.d + n])
            .sum()
    }

    /// Population in the top `levels` levels: the truncation `discarded()` misses.
    fn tail(&self, levels: usize) -> f64 {
        let k = levels.min(self.d);

        (self.d - k..self.d)
            .map(|n| self.re[n * self.d + n])
            .sum()
    }

    /// The worse of `discarded()` and `tail(EDGE)`.
    fn trunc_error(&self) -> f64 {
        f64::max(self.lost, self.tail(EDGE))
    }

    /// Moments (mean, cov) in the internal convention, for `GaussianState.from_moments`.
    fn moments(&self) -> (Vec<f64>, Vec<f64>) {
        self.raw_moments()
    }

    /// Wigner on the grid (xs, ps) as a flat float64 array, row-major with shape
    /// (len(ps), len(xs)): entry [i, j] is W(xs[j], ps[i]), the Gaussian layout.
    fn wigner<'py>(
        &self,
        py: Python<'py>,
        xs: Vec<f64>,
        ps: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        grid_step("xs", &xs)?;
        grid_step("ps", &ps)?;
        let out = py.detach(|| self.wigner_grid(&xs, &ps));

        Ok(out.into_pyarray(py))
    }

    /// Husimi Q on (xs, ps), `wigner`'s layout. In dx dp: 0 <= Q <= 1/(2 pi), attained by the
    /// vacuum (1/pi under d^2 alpha).
    fn husimi<'py>(
        &self,
        py: Python<'py>,
        xs: Vec<f64>,
        ps: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        grid_step("xs", &xs)?;
        grid_step("ps", &ps)?;

        let nx = xs.len();
        let out = py.detach(|| {
            let mut out = vec![0.0; ps.len() * nx];
            out.par_chunks_mut(nx).enumerate().for_each(|(i, row)| {
                let pp = ps[i];
                let mut cr = vec![0.0; self.d];
                let mut ci = vec![0.0; self.d];
                for (j, xx) in xs.iter().enumerate() {
                    row[j] = self.husimi_at(*xx, pp, &mut cr, &mut ci);
                }
            });

            out
        });

        Ok(out.into_pyarray(py))
    }

    /// Wigner negativity volume, integral of |W| minus 1: zero for every Gaussian
    /// state (Hudson), positive for |n>, n >= 1, odd cats and GKP. The grid is an
    /// argument: narrow windows truncate the tails, coarse steps the zero kink.
    fn negativity(&self, py: Python<'_>, xs: Vec<f64>, ps: Vec<f64>) -> PyResult<f64> {
        let dx = grid_step("xs", &xs)?;
        let dp = grid_step("ps", &ps)?;
        let total = py.detach(|| {
            self.wigner_grid(&xs, &ps)
                .iter()
                .map(|w| w.abs())
                .sum::<f64>()
                * dx
                * dp
        });

        Ok(total - 1.0)
    }

    /// Photon-number parity <(-1)^n>, exactly pi * W(0, 0), no grid needed.
    /// Negative certifies Wigner negativity; non-negative proves nothing.
    fn parity(&self) -> f64 {
        (0..self.d)
            .map(|n| {
                let s = if n % 2 == 0 { 1.0 } else { -1.0 };

                s * self.re[n * self.d + n]
            })
            .sum()
    }

    /// Purity Tr(rho^2): 1 for a pure state, 1/d for the maximally mixed one.
    fn purity(&self) -> f64 {
        self.re
            .iter()
            .zip(self.im.iter())
            .map(|(r, i)| r * r + i * i)
            .sum()
    }

    /// Eigenvalues of the density matrix, ascending.
    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let e = py.detach(|| eigvals(self.d, &self.re, &self.im));

        e.into_pyarray(py)
    }

    /// Von Neumann entropy in bits, -Tr(rho log2 rho).
    fn entropy(&self) -> f64 {
        eigvals(self.d, &self.re, &self.im)
            .iter()
            .filter(|l| **l > TINY)
            .map(|l| -l * l.log2())
            .sum()
    }

    /// Whether rho is Hermitian, unit trace and positive semidefinite within `tol`.
    fn physical(&self, tol: f64) -> PyResult<bool> {
        check_finite("tol", tol)?;

        let d = self.d;
        for i in 0..d {
            for j in 0..d {
                let gap =
                    (self.re[i * d + j] - self.re[j * d + i]).abs() + (self.im[i * d + j] + self.im[j * d + i]).abs();
                if gap > tol {
                    return Ok(false);
                }
            }
        }

        let trace: f64 = (0..d).map(|i| self.re[i * d + i]).sum();
        if (trace - 1.0).abs() > tol {
            return Ok(false);
        }

        let low = eigvals(d, &self.re, &self.im).first().copied().unwrap_or(0.0);

        Ok(low >= -tol)
    }

    /// Relative-entropy non-Gaussianity in bits, S(rho_G) - S(rho) for rho_G with
    /// this state's moments. Weaker than `negativity()`; a negative is round-off.
    fn nongauss(&self) -> f64 {
        let (_, cov) = self.raw_moments();
        let det = cov[0] * cov[3] - cov[1] * cov[2];
        let nu = det.max(0.0).sqrt();

        g_ent((nu - 0.5).max(0.0)) - self.entropy()
    }

    /// Phase-space rotation by `theta`, counterclockwise, matching the Gaussian
    /// layer's `rotate`. Diagonal in the number basis, so exact.
    fn rotate(&self, theta: f64) -> PyResult<Self> {
        check_finite("theta", theta)?;

        let d = self.d;

        // The phase depends only on m - n: 2d - 1 values, not d^2.
        let mut turn = vec![(0.0, 0.0); 2 * d - 1];
        for (i, slot) in turn.iter_mut().enumerate() {
            *slot = (theta * (i as f64 - (d as f64 - 1.0))).sin_cos();
        }

        let mut re = vec![0.0; d * d];
        let mut im = vec![0.0; d * d];
        for m in 0..d {
            for n in 0..d {
                let (s, c) = turn[m + d - 1 - n];
                let (rr, ri) = (self.re[m * d + n], self.im[m * d + n]);
                re[m * d + n] = rr * c - ri * s;
                im[m * d + n] = rr * s + ri * c;
            }
        }

        Ok(Self::build(d, re, im, self.lost))
    }

    /// Displace by beta = (dx + i dp)/sqrt(2). The truncated displacement is the
    /// exact one's top-left block, not unitary: lost trace goes to `discarded()`.
    fn displace(&self, dx: f64, dp: f64) -> PyResult<Self> {
        check_finite("dx", dx)?;
        check_finite("dp", dp)?;

        let d = self.d;
        let beta = (dx / SQRT2, dp / SQRT2);
        let nb = beta.0 * beta.0 + beta.1 * beta.1;
        let (cr, ci, _) = Self::coh_amps(beta, d);

        // D_{n+k,n} = c_k d_{n,k}(|beta|^2), D_{n,n+k} = (-1)^k conj(c_k)
        // d_{n,k}(|beta|^2): the Wigner kernel's Laguerre column, other argument.
        let mut dr = vec![0.0; d * d];
        let mut di = vec![0.0; d * d];
        for k in 0..d {
            let flip = if k % 2 == 0 { 1.0 } else { -1.0 };
            let mut prev = 0.0;
            let mut cur = 1.0;
            for n in 0..(d - k) {
                dr[(n + k) * d + n] = cr[k] * cur;
                di[(n + k) * d + n] = ci[k] * cur;
                dr[n * d + n + k] = flip * cr[k] * cur;
                di[n * d + n + k] = -flip * ci[k] * cur;
                let next = lag_next(n, k, nb, cur, prev);
                prev = cur;
                cur = next;
            }
        }

        // rho' = D rho D^dagger.
        let mut tr = vec![0.0; d * d];
        let mut ti = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut ar = 0.0;
                let mut ai = 0.0;
                for k in 0..d {
                    let (xr, xi) = (dr[i * d + k], di[i * d + k]);
                    let (yr, yi) = (self.re[k * d + j], self.im[k * d + j]);
                    ar += xr * yr - xi * yi;
                    ai += xr * yi + xi * yr;
                }
                tr[i * d + j] = ar;
                ti[i * d + j] = ai;
            }
        }

        let mut re = vec![0.0; d * d];
        let mut im = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut ar = 0.0;
                let mut ai = 0.0;
                for k in 0..d {
                    let (xr, xi) = (tr[i * d + k], ti[i * d + k]);
                    let (yr, yi) = (dr[j * d + k], -di[j * d + k]);
                    ar += xr * yr - xi * yi;
                    ai += xr * yi + xi * yr;
                }
                re[i * d + j] = ar;
                im[i * d + j] = ai;
            }
        }

        let trace: f64 = (0..d).map(|i| re[i * d + i]).sum();
        if !(trace > 0.0) {
            return Err(PyValueError::new_err(
                "the displaced state fell entirely outside the cutoff: raise it",
            ));
        }
        for x in re.iter_mut().chain(im.iter_mut()) {
            *x /= trace;
        }

        Ok(Self::build(d, re, im, 1.0 - (1.0 - self.lost) * trace))
    }

    /// Pure-loss channel of transmittance `eta`, through its number-basis Kraus
    /// operators. Trace-preserving inside the cutoff: population only moves down.
    fn loss(&self, eta: f64) -> PyResult<Self> {
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(PyValueError::new_err(format!("eta must be in [0, 1], got {eta}")));
        }

        let d = self.d;
        let lf = ln_fact(d);
        let l1 = (1.0 - eta).ln();
        let le = eta.ln();

        // log weight = f(m, k) + f(n, k). The guards keep ln(1-eta), ln(eta) = -inf off a zero.
        let mut w = vec![0.0; d * d];
        for m in 0..d {
            for k in 0..(d - m) {
                let mut lw = 0.5 * (lf[m + k] - lf[m] - lf[k]);
                if k > 0 {
                    lw += 0.5 * (k as f64) * l1;
                }
                if m > 0 {
                    lw += 0.5 * (m as f64) * le;
                }
                w[m * d + k] = lw.exp();
            }
        }

        let mut re = vec![0.0; d * d];
        let mut im = vec![0.0; d * d];
        for m in 0..d {
            for n in 0..d {
                let mut ar = 0.0;
                let mut ai = 0.0;
                for k in 0..(d - m.max(n)) {
                    let wk = w[m * d + k] * w[n * d + k];
                    ar += wk * self.re[(m + k) * d + n + k];
                    ai += wk * self.im[(m + k) * d + n + k];
                }
                re[m * d + n] = ar;
                im[m * d + n] = ai;
            }
        }

        Ok(Self::build(d, re, im, self.lost))
    }
}
