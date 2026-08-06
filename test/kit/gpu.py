from MDR import Question

from qkd import _core

READY = _core.gpu_ready()

PROBE = _core.gpu_probe()


class NeedsGpu(Question):
    """
    A question with nothing to assert without a compute adapter. No adapter is a skip,
    never a failure.
    """

    def setUp(self):
        if not READY:
            self.skipTest(f"no gpu compute path ({PROBE})")
