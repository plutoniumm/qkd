import numpy as np

import qkd as q

# Wheel smoke test, the last gate before upload: one check of each kind. A
# manylinux container has no adapter, so a pass there says nothing about the
# GPU path.
#
# numpy and the built wheel are the only imports available here -- this runs
# inside the manylinux containers, where nothing else is installed.

TOL = 1e-4


def close(got, want, tol=TOL):
    return abs(float(got) - float(want)) <= tol


assert q.__file__ is not None, "namespace package: the wheel has no __init__.py"

absent = [n for n in q.__all__ if not hasattr(q, n)]

assert not absent, f"qkd.__all__ promises names the wheel lacks: {absent}"

name, precision, device = q.backend_info()

assert name in ("cpu", "gpu"), name

assert precision in ("f32", "f64"), precision

assert device, "backend reported no device"

# Dispatch asks for a precision, not a device. WGSL has no f64.
assert q.supports(precision), f"{name} does not support its own {precision}"

if name == "gpu":
    assert precision == "f32", precision

    assert not q.supports("f64"), "gpu backend must never claim f64"
else:
    assert q.supports("f64") and q.supports("f32"), "cpu must serve both"

# hbar = 1, x = (a + a^dag)/sqrt(2): vacuum variance 1/2 in both quadratures.
vac = q.gaussian.Vacuum(1)

assert vac.n_modes == 1, vac.n_modes

assert np.allclose(vac.cov, 0.5 * np.eye(2)), vac.cov

assert close(vac.purity(), 1.0), vac.purity()

assert close(vac.entropy(), 0.0), vac.entropy()

# rust-numpy resolves the NumPy C-API through the _ARRAY_API capsule at
# runtime, so one array-returning call needs a built wheel meeting a real NumPy.
draws = vac.homodyne(0, angle=0.0, shots=64, seed=7)

assert isinstance(draws, np.ndarray), type(draws)

assert draws.dtype == np.float64, draws.dtype

assert draws.shape == (64,), draws.shape

assert np.isfinite(draws).all(), draws

# xi at the channel input.
lossy = q.gaussian.Squeezed(0.8).thermal_loss(0, 0.3, xi=0.05, ref="input")
cov = lossy.cov

assert lossy.physical(), cov

assert cov[0, 0] * cov[1, 1] >= 0.25 - 1e-12, cov

assert lossy.entropy() > 0.0, "a lossy state cannot stay pure"

# Lodewyck 2007 (arXiv:0706.4255), 25 km, homodyne, trusted detector.
res = q.Link(
    modulation=q.GaussianModulation(v_a=18.5),
    channel=q.Channel(T=0.302, xi=0.005, ref="input"),
    bob=q.Bob(detector=q.Homodyne(eta=0.606, v_el=0.041, trusted=True)),
    security=q.Asymptotic(beta=0.898),
).run()

assert close(res.i_ab, 1.0436), res.i_ab

assert close(res.chi_be, 0.9020), res.chi_be

assert close(res.key_rate, 0.0352), res.key_rate

print(
    f"qkd {q.__version__} smoke OK: "
    f"gpu_feature={q.__gpu__}, backend={name}/{precision} on {device}, "
    f"vacuum=1/2, lodewyck K={res.key_rate:.4f} bit/symbol"
)
