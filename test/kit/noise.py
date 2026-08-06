import warnings

from qkd import budget


def assembled(v_a, t, v_err=2e-3, adc_bits=12):
    """
    The metro noise budget -- 16-bit DAC, -155 dBc/Hz laser, 100 MHz bandwidth -- with
    modulation, channel and ADC resolution left to the caller.
    """

    return budget.assemble(
        v_a=v_a,
        t=t,
        v_err=v_err,
        rin=-155.0,
        dac_bits=16,
        adc_bits=adc_bits,
        bandwidth=100e6,
    )


def quiet(v_a, v_err, xi=0.0, form="estimator"):
    """
    budget.phase with PhaseDomainWarning muted. test_phase_domain_warns checks the warning.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", budget.PhaseDomainWarning)

        return budget.phase(v_a, v_err, xi, form=form)
