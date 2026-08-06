// The two fixed-count bracket searches and the one divergence more than one engine inverts.
// Kept out on purpose: argument checks, phase wrapping and the click law (`crate::std`); any
// Cholesky (`sdp.rs`, `herm.rs`, `lp.rs` each carry their own, per their headers); any Poisson
// weight (`click::poisson_tail`, `rrdps::poisson_cdf`, `decoy::tau`, `sarg::tau` and
// `pairing.rs`'s inline `mu e^-mu` are five rounding orders in five engines); and two bisections
// (`ekert_threshold`'s objective is fallible, `dmcs::gh_bisect` breaks on no progress with the
// orientation reversed).
//
// EVERY STEP COUNT IS THE CALLER'S: `b92_tolerance` halves 200 times, `rrdps_tolerance` 100.
// Below `want ~ 1e-18` one ulp takes ~118 halvings, so they differ: do not unify.

/// Golden-section ratio, for `golden` and `pairs::pair_optimum`'s own loop.
pub(crate) const PHI: f64 = 0.618_033_988_749_894_8;

/// `steps` halvings of `[lo, hi]`: `keep(mid)` true moves `lo` up, false moves `hi` down, so
/// `lo` is the last point where `keep` held and `hi` the first where it failed. Fixed count,
/// never a tolerance: a hundred halvings of a unit bracket is below f64 resolution.
pub(crate) fn bisect(lo: f64, hi: f64, steps: usize, keep: impl Fn(f64) -> bool) -> (f64, f64) {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..steps {
        let mid = 0.5 * (lo + hi);
        if keep(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo, hi)
}

/// Argmax of a unimodal `f` on `[lo, hi]` by golden section, `steps` shrinks of 0.618, midpoint
/// of what is left. Every caller brackets on a grid first.
pub(crate) fn golden(lo: f64, hi: f64, steps: usize, f: impl Fn(f64) -> f64) -> f64 {
    let (mut left, mut right) = (lo, hi);
    for _ in 0..steps {
        let m1 = right - PHI * (right - left);
        let m2 = left + PHI * (right - left);
        if f(m1) < f(m2) {
            left = m1;
        } else {
            right = m2;
        }
    }

    0.5 * (left + right)
}

/// Binary KL divergence in BITS, `D(q||p) = q log2(q/p) + (1-q) log2((1-q)/(1-p))`, the
/// Chernoff exponent `2^{-n D(k/n||p)}` every consumer inverts. Zero arguments contribute 0, not
/// NaN, so `D(0||p) = log2(1/(1-p))` and `D(1||p) = log2(1/p)` are exact and `k = n` is tight.
pub(crate) fn kl_bits(q: f64, p: f64) -> f64 {
    let hi = if q > 0.0 { q * (q / p).log2() } else { 0.0 };
    let lo = if q < 1.0 {
        (1.0 - q) * ((1.0 - q) / (1.0 - p)).log2()
    } else {
        0.0
    };

    hi + lo
}
