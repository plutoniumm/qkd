import math

LN2 = math.log(2.0)


def background(dark):
    """
    Background yield per pulse from a per-gate dark probability: two gates, composed rather
    than doubled. Mirrors Link._background.
    """

    return 1.0 - (1.0 - dark) ** 2


def band(p, n, sigmas=4.0):
    """
    A ``sigmas``-wide binomial confidence half-width on proportion ``p`` from ``n`` trials.
    """
    if n <= 0:
        return 1.0

    return sigmas * math.sqrt(max(p * (1.0 - p), 1e-12) / n)


# Five loops in test/ stay hand-rolled: expcore's `cutoff`, b92's `crossing` and dmcs's
# `reach` halve 24 to 60 times where 200 costs ~8x, 3.3x and ~9x the engine calls;
# blind.py's two bisect a boolean predicate.
def bisect(fn, lo, hi, target=0.0):
    """
    Where a monotone ``fn`` crosses ``target`` in [lo, hi]. Direction is read off the
    bracket; 200 halvings put the crossing past its last bit.
    """
    keep = fn(lo) > target

    for _ in range(200):
        mid = 0.5 * (lo + hi)

        if (fn(mid) > target) == keep:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def fitslope(xs, ys):
    """
    Least-squares slope of ``ys`` against ``xs``; callers pass logarithms already taken.
    """
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))

    return num / sum((x - mx) ** 2 for x in xs)


# Three copies stay out of this one: blind.py's h2 is that file's independent-reimplementation
# charter, keyrate.py's `entropy` is the thermal G(x) and relay.py's a von Neumann grouping.
def h2(e):
    """
    Binary entropy in bits, 0.0 at and outside the closed endpoints. Reimplemented here,
    never crate::std::h2 or qkd.std.entropy.
    """
    if e <= 0.0 or e >= 1.0:
        return 0.0

    return -e * math.log2(e) - (1.0 - e) * math.log2(1.0 - e)


# Pirandola, Laurenza, Ottaviani & Banchi, Nature Communications 8, 15043 (2017): the
# repeaterless secret-key capacity of a pure-loss channel, in bits per channel use.
# Never -log2(1 - eta): in f64 that form is 11% high at 160 dB and exactly 0.0 by 176 dB.
# Measured by test/capacity.py::test_capacity_form.
def plob(eta):
    """
    The repeaterless capacity, in the only form that survives high loss.
    """

    return -math.log1p(-eta) / LN2


def poisson(mu, n):
    """
    Poisson weight of n photons at mean mu.
    """
    if mu <= 0.0:
        return 1.0 if n == 0 else 0.0

    return math.exp(-mu) * mu**n / math.factorial(n)


def untrusted(t, xi, eta, vel, mu):
    """
    (T, xi) at the channel input once Bob's receiver is Eve's: (eta*T, xi + mu*v_el/(eta*T)),
    mu = 1 for a homodyne quadrature and 2 for a heterodyne pair. Charging v_el without
    attenuating T is the referring-plane error.
    """
    total = eta * t

    return total, xi + mu * vel / total
