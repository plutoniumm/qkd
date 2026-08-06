import sys

from MDB import Suite, Trial, load, spin, stamp

from qkd import _core

# Priced against the bare PyO3 call cost (bench_ffi). Every case but the Basis
# sweep loops internally, so the Rate column is calls/s there and pulse/s for
# Basis, whose rows are one run_basis call each.

# guide/link.md defaults at 20 km of fibre: heterodyne, imperfect receiver.
CV = (4.0, 0.5, 0.01, 0.6, 0.1, 0.95, False, False)

# Weak-coherent decoy source, two decoys, at the rates a 50 km link produces.
DECOY = (0.5, 0.1, 0.0, 0.03, 0.03, 0.006, 0.04, 1e-5, 0.5)

# Unequal arms, uncorrelated environment: the branch cvmdi_rate takes when the
# symmetric closed form does not apply.
RELAY = (0.8, 0.4, 1.0, 1.0, 0.0, 0.0)

ALPHA = 0.5

# The source DECOY's observables came from, sampled rather than inverted: 50 km
# at 0.2 dB/km into an 80% detector, whose signal gain is 0.039 against the 0.03
# that tuple carries. Order: intensities (DECOY[:3]), probabilities, t, eta,
# dark, misalign, sift -- run_basis takes n, seed and the grain around them.
BASIS = (DECOY[:3], (0.6, 0.25, 0.15), 0.1, 0.8, 1e-6, 0.03, 0.5)


class Closed(Trial):
    warmup = 2
    runs = 5

    def bench_ffi(self):
        """
        supports("f64"), a PyO3 round trip that computes nothing: the floor
        every other row here must be read against.
        """

        return spin(lambda: _core.supports("f64"))

    def bench_decoy(self):
        """
        decoy_bounds: the two-decoy single-photon yield and error bounds.
        """

        return spin(lambda: _core.decoy_bounds(*DECOY))

    def bench_gain(self):
        """
        decoy_gain, the forward map from mean photon number to gain and QBER.
        """

        return spin(lambda: _core.decoy_gain(0.5, 0.1, 1e-6, 0.015))

    def bench_bb84(self):
        """
        bb84_rate: the GLLP rate from sifted gain and single-photon bounds.
        """

        return spin(lambda: _core.bb84_rate(0.5, 0.03, 0.03, 0.02, 0.01, 1.16))

    def bench_cow(self):
        """
        cow_rate: the coherent-one-way rate from bit and phase error rates.
        """

        return spin(lambda: _core.cow_rate(0.05, 0.01, 0.05, 1.1))

    def bench_dps(self):
        """
        dps_rate: the differential-phase-shift rate from click rate and QBER.
        """

        return spin(lambda: _core.dps_rate(0.01, 0.03, 0.2, 1.1))

    def bench_cv(self):
        """
        cv_rate: the asymptotic Gaussian-modulation rate, untrusted receiver.
        """

        return spin(lambda: _core.cv_rate(*CV))

    def bench_finite(self):
        """
        cv_finite: the same rate with finite-size confidence bounds folded in.
        """

        return spin(lambda: _core.cv_finite(*CV, 1e9, 0.5, 1e-10, 1e-10, 1e-10))

    def bench_bounds(self):
        """
        cv_bounds: the parameter-estimation confidence interval on T and xi.
        """

        return spin(lambda: _core.cv_bounds(0.5, 1.2, 1e6, 4.0, 1e-10))


class Relay(Trial):
    warmup = 2
    runs = 5

    def setup(self):
        self.chi = _core.cvmdi_noise(*RELAY)

    def bench_noise(self):
        """
        cvmdi_noise: the relay's equivalent excess noise from the arm data.
        """

        return spin(lambda: _core.cvmdi_noise(*RELAY))

    def bench_rate(self):
        """
        cvmdi_rate: the asymptotic relay rate, a closed form in three scalars.
        """

        return spin(lambda: _core.cvmdi_rate(RELAY[0], RELAY[1], self.chi))

    def bench_cov(self):
        """
        cvmdi_cov: the post-relay 4x4, routed through the Gaussian layer's
        beamsplitter and conditionings rather than evaluated as a formula.
        """

        return spin(lambda: _core.cvmdi_cov(20.0, *RELAY), 2000)

    def bench_point(self):
        """
        cvmdi_point: cvmdi_cov plus the symplectic spectrum and the Holevo
        bound, which is the finite-modulation relay rate.
        """

        return spin(lambda: _core.cvmdi_point(20.0, *RELAY, 0.95), 2000)


class Basis(Trial):
    """
    The purest uniforms-bound fold in the crate: two Threefry blocks per pulse
    and a body of comparisons and u64 increments, no libm at all. Against
    bench_ffi it reads the RNG alone, where every other sampler here mixes
    transcendentals into the same measurement.
    """

    warmup = 1
    runs = 3

    def bench_1e5(self):
        """
        A hundred thousand pulses: small enough that rayon ramp-up still shows.
        """
        _core.run_basis(int(1e5), 7, *BASIS, 0)

        return (1e5, "pulse")

    def bench_1e6(self):
        """
        One million pulses, the default Link.run() size.
        """
        _core.run_basis(int(1e6), 7, *BASIS, 0)

        return (1e6, "pulse")

    def bench_1e7(self):
        """
        Ten million pulses.
        """
        _core.run_basis(int(1e7), 7, *BASIS, 0)

        return (1e7, "pulse")


class Gaussian(Trial):
    """
    Every gate here clones the state, applies a K x K map across the whole
    2n x 2n covariance and revalidates, so the map's share of the call falls as
    the state grows. bench_bs is K = 4; the two squeeze rows are K = 2 at four
    modes and at one. bench_moments and bench_condition are the controls that
    never reach apply_map.
    """

    warmup = 2
    runs = 5

    def setup(self):
        base = _core.GaussianState.vacuum(4)
        self.four = base.squeeze(0, 0.4).bs(0, 1, 0.5).squeeze(2, 0.2).bs(2, 3, 0.3)
        self.one = _core.GaussianState.vacuum(1).squeeze(0, 0.3)
        self.mean = list(self.four.mean())
        self.cov = list(self.four.cov())

    def bench_bs(self):
        """
        bs on a four-mode state: the K = 4 map over the 8x8, then a bona fide
        check on the result.
        """

        return spin(lambda: self.four.bs(0, 2, 0.5))

    def bench_squeeze4(self):
        """
        squeeze on the same four-mode state: a K = 2 map over the same 8x8.
        """

        return spin(lambda: self.four.squeeze(1, 0.3))

    def bench_squeeze1(self):
        """
        squeeze on a one-mode state, the 2x2 where the map is most of the call.
        """

        return spin(lambda: self.one.squeeze(0, 0.3))

    def bench_moments(self):
        """
        A control: from_moments on the same 8x8 checks shape, finiteness,
        symmetry and bona fide but applies no map. It is the only row reaching
        bona_fide, so a Cholesky change moves it legitimately.
        """

        return spin(lambda: _core.GaussianState.from_moments(self.mean, self.cov))

    def bench_condition(self):
        """
        A control: condition 4 to 3 writes a fresh smaller covariance by Schur
        complement and applies no map, so apply_map cannot move it either.
        """

        return spin(lambda: self.four.condition(3, 0.0, 0.1))


class Discrete(Trial):
    """
    dm_info's cost splits into an M-independent part (2048 Gauss-Hermite node
    pairs) and an M-dependent one. Sweeping M separates them by a straight-line
    fit, whose intercept is what a smaller node count would buy.
    """

    warmup = 2
    runs = 5

    def bench_states(self):
        """
        dm_states at M = 4: the constellation itself, one sin_cos per symbol.
        """

        return spin(lambda: _core.dm_states(4, ALPHA))

    def bench_moments(self):
        """
        dm_moments at M = 4: second moments, O(M) with an exponential each.
        """

        return spin(lambda: _core.dm_moments(4, ALPHA), 5000)

    def bench_holevo(self):
        """
        dm_holevo: the Holevo bound from a given correlation, closed form.
        """

        return spin(lambda: _core.dm_holevo(0.5, 0.5, 0.01, 0.5, 2.0, 0.95))

    def bench_info3(self):
        """
        dm_info at M = 3, the smallest legal constellation.
        """

        return spin(lambda: _core.dm_info(3, ALPHA, 0.5, 0.01), 1000)

    def bench_info4(self):
        """
        dm_info at M = 4 (QPSK).
        """

        return spin(lambda: _core.dm_info(4, ALPHA, 0.5, 0.01), 1000)

    def bench_info8(self):
        """
        dm_info at M = 8.
        """

        return spin(lambda: _core.dm_info(8, ALPHA, 0.5, 0.01), 1000)

    def bench_info16(self):
        """
        dm_info at M = 16.
        """

        return spin(lambda: _core.dm_info(16, ALPHA, 0.5, 0.01), 500)

    def bench_info32(self):
        """
        dm_info at M = 32.
        """

        return spin(lambda: _core.dm_info(32, ALPHA, 0.5, 0.01), 500)

    def bench_info64(self):
        """
        dm_info at M = 64, the largest constellation the module accepts.
        """

        return spin(lambda: _core.dm_info(64, ALPHA, 0.5, 0.01), 200)

    def bench_rate4(self):
        """
        dm_rate at M = 4: dm_info plus the moments and the Holevo bound.
        """

        return spin(lambda: _core.dm_rate(4, ALPHA, 0.5, 0.01, 0.6, 0.1, 0.95), 1000)

    def bench_rate64(self):
        """
        dm_rate at M = 64.
        """

        return spin(lambda: _core.dm_rate(64, ALPHA, 0.5, 0.01, 0.6, 0.1, 0.95), 200)


if __name__ == "__main__":
    meta = stamp()
    rc = 0
    rc |= Suite(
        "Closed-form rates",
        "Every scalar rate path in the crate, against the bare PyO3 call cost",
        "protocols_closed.md",
        meta=meta,
    ).run(load(Closed))
    rc |= Suite(
        "CV-MDI relay",
        "The relay's closed form against its covariance route through the Gaussian layer",
        "protocols_relay.md",
        meta=meta,
    ).run(load(Relay))
    rc |= Suite(
        "Basis keying",
        "run_basis swept in the pulse count: the uniforms-bound sampler, two "
        "Threefry blocks per pulse and no transcendental in the fold body",
        "protocols_basis.md",
        meta=meta,
    ).run(load(Basis))
    rc |= Suite(
        "Gaussian layer",
        "The symplectic gates against two controls that never reach apply_map, "
        "so a change confined to the map has somewhere to read as zero",
        "protocols_gaussian.md",
        meta=meta,
    ).run(load(Gaussian))
    rc |= Suite(
        "Discrete modulation",
        "dm_info and dm_rate swept in the constellation size M, to separate the "
        "Gauss-Hermite node cost from the per-symbol cost",
        "protocols_dmcs.md",
        meta=meta,
    ).run(load(Discrete))
    sys.exit(rc)
