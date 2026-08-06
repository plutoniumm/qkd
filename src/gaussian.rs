use numpy::{IntoPyArray, PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::std::{check_finite, check_nonneg, check_prob, sqrt0};

// n-mode Gaussian states. hbar = 1, vacuum variance 1/2 per quadrature, xpxp ordering. n is
// single-digit, so nested loops beat a BLAS handoff; f64 on the CPU.

/// SplitMix64 (Steele et al. 2014 finalizer). Shot noise, not cryptography.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// Uniform in (0, 1]; the open end keeps ln(u) finite in Box-Muller.
    fn uniform(&mut self) -> f64 {
        (((self.next_word() >> 11) + 1) as f64) / 9_007_199_254_740_992.0
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let th = std::f64::consts::TAU * self.uniform();

        (r * th.cos(), r * th.sin())
    }
}

/// `m v`, each row accumulated in column order -- the order `apply_map` contracts.
fn matvec<const K: usize>(m: &[[f64; K]; K], v: &[f64; K]) -> [f64; K] {
    std::array::from_fn(|i| {
        let mut acc = 0.0;
        for l in 0..K {
            acc += m[i][l] * v[l];
        }

        acc
    })
}

/// ABSOLUTE: the scale of V + i*Omega/2 is the 1/2 of its Omega/2 entries, not the state's
/// brightness. Relative 1e-9 would accept V_xx = 5e-97, V_pp = 1e6.
const BONA_TOL: f64 = 1e-9;

/// V + i*Omega/2 >= 0 by Cholesky of the Hermitian matrix shifted by +tol, which absorbs a pure
/// state's exact zero. Omega is block diagonal in xpxp, so only within-mode entries are imaginary.
fn bona_fide(n: usize, cov: &[f64]) -> bool {
    let dim = 2 * n;
    let tol = BONA_TOL;
    let mut pool = vec![0.0; 4 * dim * dim];
    let (ar, rest) = pool.split_at_mut(dim * dim);
    let (ai, rest) = rest.split_at_mut(dim * dim);
    let (lr, li) = rest.split_at_mut(dim * dim);
    ar.copy_from_slice(cov);
    for i in 0..dim {
        ar[i * dim + i] += tol;
    }
    for k in 0..n {
        ai[(2 * k) * dim + 2 * k + 1] = 0.5;
        ai[(2 * k + 1) * dim + 2 * k] = -0.5;
    }

    for j in 0..dim {
        let mut piv = ar[j * dim + j];
        for k in 0..j {
            piv -= lr[j * dim + k] * lr[j * dim + k] + li[j * dim + k] * li[j * dim + k];
        }
        if !(piv > 0.0) {
            return false;
        }

        let root = piv.sqrt();
        lr[j * dim + j] = root;
        for i in (j + 1)..dim {
            // s = A[i][j] - sum_k L[i][k] * conj(L[j][k]).
            let mut sr = ar[i * dim + j];
            let mut si = ai[i * dim + j];
            for k in 0..j {
                sr -= lr[i * dim + k] * lr[j * dim + k] + li[i * dim + k] * li[j * dim + k];
                si -= li[i * dim + k] * lr[j * dim + k] - lr[i * dim + k] * li[j * dim + k];
            }
            lr[i * dim + j] = sr / root;
            li[i * dim + j] = si / root;
        }
    }

    true
}

/// Mean 2n, covariance 2n x 2n row-major, xpxp. hbar = 1, vacuum 1/2 -- not SNU; the SNU layer
/// doubles variances at its boundary.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct GaussianState {
    n: usize,
    mean: Vec<f64>,
    cov: Vec<f64>,
}

impl GaussianState {
    fn check_mode(&self, mode: usize) -> PyResult<()> {
        if mode < self.n {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "mode {mode} out of range for a {}-mode state",
                self.n
            )))
        }
    }

    /// Flat row-major, for Rust callers; Python reaches it through `cov_array`.
    pub(crate) fn cov(&self) -> Vec<f64> {
        self.cov.clone()
    }

    /// mean -> S mean, cov -> S cov S^T, S the identity extended by `m` on `qs`. Whole rows then
    /// whole columns, so cross terms reach untouched modes. THE SUMMATION ORDER IS A CONTRACT.
    /// A map whose shape disagrees with `qs` is a COMPILE error; no runtime guard is written.
    fn apply_map<const K: usize>(&mut self, m: &[[f64; K]; K], qs: &[usize; K]) {
        let dim = 2 * self.n;

        let old: [f64; K] = std::array::from_fn(|l| self.mean[qs[l]]);
        let new = matvec(m, &old);
        for i in 0..K {
            self.mean[qs[i]] = new[i];
        }

        for col in 0..dim {
            let tmp: [f64; K] = std::array::from_fn(|l| self.cov[qs[l] * dim + col]);
            let new = matvec(m, &tmp);
            for i in 0..K {
                self.cov[qs[i] * dim + col] = new[i];
            }
        }

        for row in 0..dim {
            let tmp: [f64; K] = std::array::from_fn(|l| self.cov[row * dim + qs[l]]);
            let new = matvec(m, &tmp);
            for i in 0..K {
                self.cov[row * dim + qs[i]] = new[i];
            }
        }
    }

    /// Mean and variance of u = x*cos(angle) + p*sin(angle). Non-finite is an error, not a
    /// zero: `sqrt0` reads NaN as 0, leaving `homodyne` noiseless.
    fn marginal(&self, mode: usize, angle: f64) -> PyResult<(f64, f64)> {
        let dim = 2 * self.n;
        let (ix, ip) = (2 * mode, 2 * mode + 1);
        let (ux, up) = (angle.cos(), angle.sin());
        let mu = ux * self.mean[ix] + up * self.mean[ip];
        let var = ux * ux * self.cov[ix * dim + ix]
            + 2.0 * ux * up * self.cov[ix * dim + ip]
            + up * up * self.cov[ip * dim + ip];

        if !(mu.is_finite() && var.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "the marginal of mode {mode} at angle {angle} is not finite (mean {mu}, \
                 variance {var})"
            )));
        }

        Ok((mu, var))
    }

    /// Every state handed back must be one `from_moments` would re-admit: e^(2r) leaves f64 at
    /// r ~ 355 and diag(e^-r, e^+r) at r ~ 710, where `apply_map` writes 0*inf.
    fn check_moments(&self, op: &str) -> PyResult<()> {
        for (i, x) in self.mean.iter().chain(self.cov.iter()).enumerate() {
            if !x.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{op} produced a non-finite state: moment {i} is {x}, which \
                     from_moments would refuse"
                )));
            }
        }

        Ok(())
    }
}

#[pymethods]
impl GaussianState {
    /// The n-mode vacuum: zero mean, covariance I/2.
    #[staticmethod]
    fn vacuum(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("a state needs at least one mode"));
        }

        let dim = 2 * n;
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 0.5;
        }

        Ok(Self {
            n,
            mean: vec![0.0; dim],
            cov,
        })
    }

    /// Internal units (hbar = 1, vacuum 1/2, xpxp). Every derived state returns through here.
    #[staticmethod]
    pub(crate) fn from_moments(mean: Vec<f64>, cov: Vec<f64>) -> PyResult<Self> {
        let dim = mean.len();
        if dim == 0 || dim % 2 != 0 {
            return Err(PyValueError::new_err(format!(
                "mean must have an even, non-zero length (2 per mode), got {dim}"
            )));
        }
        if cov.len() != dim * dim {
            return Err(PyValueError::new_err(format!(
                "cov must be {dim}x{dim} = {} entries for a {dim}-quadrature mean, got {}",
                dim * dim,
                cov.len()
            )));
        }
        for (i, x) in mean.iter().chain(cov.iter()).enumerate() {
            if !x.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "moments must be finite, entry {i} is {x}"
                )));
            }
        }

        // RELATIVE where BONA_TOL is absolute: S V S^T comes back asymmetric in proportion to
        // max|V|.
        let scale = cov.iter().fold(1.0, |a: f64, x| a.max(x.abs()));
        for i in 0..dim {
            for j in (i + 1)..dim {
                let gap = (cov[i * dim + j] - cov[j * dim + i]).abs();
                if gap > 1e-9 * scale {
                    return Err(PyValueError::new_err(format!(
                        "cov must be symmetric, entries ({i}, {j}) differ by {gap:.3e}"
                    )));
                }
            }
        }

        let n = dim / 2;
        if !bona_fide(n, &cov) {
            return Err(PyValueError::new_err(
                "cov violates the bona fide condition V + i*Omega/2 >= 0",
            ));
        }

        Ok(Self { n, mean, cov })
    }

    /// Single-mode thermal state: covariance (nbar + 1/2) I, the vacuum at nbar = 0.
    #[staticmethod]
    fn thermal(nbar: f64) -> PyResult<Self> {
        check_nonneg("nbar", nbar)?;

        let mut s = Self::vacuum(1)?;
        s.cov[0] = nbar + 0.5;
        s.cov[3] = nbar + 0.5;

        Ok(s)
    }

    /// Two-mode squeezed vacuum: blocks cosh(2r)/2 I2, off-diagonal sinh(2r)/2 diag(1, -1). The
    /// f64 gap e^(-2r) of the purity cosh(2r)^2 - sinh(2r)^2 = 1 dies at r ~ 9.2 (~80 dB).
    #[staticmethod]
    fn epr(r: f64) -> PyResult<Self> {
        check_finite("r", r)?;

        let ch = (2.0 * r).cosh() / 2.0;
        let sh = (2.0 * r).sinh() / 2.0;
        if !(ch.is_finite() && ch - sh.abs() > f64::EPSILON * ch) {
            return Err(PyValueError::new_err(format!(
                "epr r = {r} is past the two-mode squeezing f64 can hold: cosh(2r) and sinh(2r) \
                 no longer differ by a representable amount"
            )));
        }

        let mut cov = vec![0.0; 16];
        for i in 0..4 {
            cov[i * 4 + i] = ch;
        }
        cov[2] = sh;
        cov[8] = sh;
        cov[7] = -sh;
        cov[13] = -sh;

        Self::from_moments(vec![0.0; 4], cov)
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.n
    }

    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.mean.to_pyarray(py)
    }

    /// FLAT 2n x 2n row-major; the caller reshapes. A fresh buffer.
    #[pyo3(name = "cov")]
    fn cov_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.cov.to_pyarray(py)
    }

    fn displace(&self, mode: usize, dx: f64, dp: f64) -> PyResult<Self> {
        self.check_mode(mode)?;
        check_finite("dx", dx)?;
        check_finite("dp", dp)?;

        let mut s = self.clone();
        s.mean[2 * mode] += dx;
        s.mean[2 * mode + 1] += dp;
        s.check_moments("displace")?;

        Ok(s)
    }

    /// diag(e^-r, e^+r), x squeezed for r > 0.
    fn squeeze(&self, mode: usize, r: f64) -> PyResult<Self> {
        self.check_mode(mode)?;
        check_finite("r", r)?;

        let mut s = self.clone();
        let m = [[(-r).exp(), 0.0], [0.0, r.exp()]];
        s.apply_map(&m, &[2 * mode, 2 * mode + 1]);
        s.check_moments("squeeze")?;

        Ok(s)
    }

    fn rotate(&self, mode: usize, theta: f64) -> PyResult<Self> {
        self.check_mode(mode)?;
        check_finite("theta", theta)?;

        let mut s = self.clone();
        let (sn, cs) = theta.sin_cos();
        let m = [[cs, -sn], [sn, cs]];
        s.apply_map(&m, &[2 * mode, 2 * mode + 1]);
        s.check_moments("rotate")?;

        Ok(s)
    }

    /// [[sqrt(t) I2, sqrt(1-t) I2], [-sqrt(1-t) I2, sqrt(t) I2]] on (x1, p1, x2, p2).
    pub(crate) fn bs(&self, m1: usize, m2: usize, t: f64) -> PyResult<Self> {
        self.check_mode(m1)?;
        self.check_mode(m2)?;
        if m1 == m2 {
            return Err(PyValueError::new_err(format!(
                "beamsplitter needs two distinct modes, got {m1} twice"
            )));
        }
        check_prob("t", t)?;

        let mut s = self.clone();
        let ct = t.sqrt();
        let st = (1.0 - t).sqrt();
        let m = [
            [ct, 0.0, st, 0.0],
            [0.0, ct, 0.0, st],
            [-st, 0.0, ct, 0.0],
            [0.0, -st, 0.0, ct],
        ];
        s.apply_map(&m, &[2 * m1, 2 * m1 + 1, 2 * m2, 2 * m2 + 1]);
        s.check_moments("bs")?;

        Ok(s)
    }

    /// Thermal-loss channel; `bs` against vacuum is pure loss only. Quadratures scale by sqrt(t),
    /// the diagonal gains (1 - t)/2 plus t*xi/2 (xi at the channel input) or xi/2 (at Bob's
    /// plane). xi is SNU against a 1/2 vacuum, hence /2.
    fn thermal_loss(&self, mode: usize, t: f64, xi: f64, input_ref: bool) -> PyResult<Self> {
        self.check_mode(mode)?;
        check_prob("t", t)?;
        check_nonneg("xi", xi)?;

        let mut s = self.clone();
        let root = t.sqrt();
        s.apply_map(&[[root, 0.0], [0.0, root]], &[2 * mode, 2 * mode + 1]);

        let add = (1.0 - t) * 0.5 + if input_ref { t * xi * 0.5 } else { xi * 0.5 };
        let dim = 2 * s.n;
        s.cov[(2 * mode) * dim + 2 * mode] += add;
        s.cov[(2 * mode + 1) * dim + (2 * mode + 1)] += add;
        s.check_moments("thermal_loss")?;

        Ok(s)
    }

    /// Samples of the quadrature at `angle` (0 = x, pi/2 = p), deterministic in `seed`. The
    /// state is not conditioned; see `condition`.
    fn homodyne<'py>(
        &self,
        py: Python<'py>,
        mode: usize,
        angle: f64,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_mode(mode)?;
        check_finite("angle", angle)?;

        let (mu, var) = self.marginal(mode, angle)?;
        let sd = sqrt0(var);
        let mut rng = SplitMix::new(seed);
        let mut out = Vec::with_capacity(shots);
        while out.len() < shots {
            let (a, b) = rng.normal_pair();
            out.push(mu + sd * a);
            if out.len() < shots {
                out.push(mu + sd * b);
            }
        }

        Ok(out.into_pyarray(py))
    }

    /// `shots` (x, p) pairs flattened, from the Husimi covariance: the mode's block plus one
    /// balanced-splitter vacuum unit (1/2 here).
    fn heterodyne<'py>(
        &self,
        py: Python<'py>,
        mode: usize,
        shots: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.check_mode(mode)?;

        let dim = 2 * self.n;
        let (ix, ip) = (2 * mode, 2 * mode + 1);
        let s11 = self.cov[ix * dim + ix] + 0.5;
        let s12 = self.cov[ix * dim + ip];
        let s22 = self.cov[ip * dim + ip] + 0.5;
        if !(s11.is_finite() && s12.is_finite() && s22.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "the Husimi covariance of mode {mode} is not finite"
            )));
        }

        let l11 = sqrt0(s11);
        let l21 = if l11 > 0.0 { s12 / l11 } else { 0.0 };
        let l22 = sqrt0(s22 - l21 * l21);

        let (mx, mp) = (self.mean[ix], self.mean[ip]);
        let mut rng = SplitMix::new(seed);
        let mut out = Vec::with_capacity(2 * shots);
        for _ in 0..shots {
            let (a, b) = rng.normal_pair();
            out.push(mx + l11 * a);
            out.push(mp + l21 * a + l22 * b);
        }

        Ok(out.into_pyarray(py))
    }

    /// Condition on a homodyne outcome at `angle`: mean_r += c_r (outcome - mu)/v, V_{rr'} -=
    /// c_r c_{r'}/v, v the measured variance, c_r its covariance with quadrature r. Leaves n-1
    /// modes.
    pub(crate) fn condition(&self, mode: usize, angle: f64, outcome: f64) -> PyResult<Self> {
        self.check_mode(mode)?;
        check_finite("angle", angle)?;
        check_finite("outcome", outcome)?;
        if self.n == 1 {
            return Err(PyValueError::new_err(
                "conditioning the only mode of a 1-mode state would leave a zero-mode object; \
                 use homodyne to sample it without conditioning",
            ));
        }

        // Not an accuracy floor: catches the v <= 0 BONA_TOL admits; must stay <= BONA_TOL.
        let (mu, v) = self.marginal(mode, angle)?;
        if v < 1e-12 {
            return Err(PyValueError::new_err(format!(
                "measured quadrature variance {v:.3e} is too small to condition on"
            )));
        }

        let dim = 2 * self.n;
        let (ix, ip) = (2 * mode, 2 * mode + 1);
        let (ux, up) = (angle.cos(), angle.sin());
        let keep: Vec<usize> = (0..dim).filter(|&q| q != ix && q != ip).collect();
        let c: Vec<f64> = keep
            .iter()
            .map(|&r| ux * self.cov[r * dim + ix] + up * self.cov[r * dim + ip])
            .collect();

        let nd = dim - 2;
        let gain = (outcome - mu) / v;
        let mut mean = Vec::with_capacity(nd);
        for (a, &r) in keep.iter().enumerate() {
            mean.push(self.mean[r] + c[a] * gain);
        }
        let mut cov = vec![0.0; nd * nd];
        for (a, &r) in keep.iter().enumerate() {
            for (b, &rp) in keep.iter().enumerate() {
                cov[a * nd + b] = self.cov[r * dim + rp] - c[a] * c[b] / v;
            }
        }

        let out = Self {
            n: self.n - 1,
            mean,
            cov,
        };
        out.check_moments("condition")?;

        Ok(out)
    }
}
