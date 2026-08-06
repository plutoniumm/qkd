from MDR import Question


class Guarded(Question):
    """
    An engine's argument domain: one good argument list, and per slot the values refused.
    """

    def assertBad(self, needle, fn, args, exc=ValueError, msg=None):
        """
        ``fn(*args)`` raises ``exc`` naming the bad parameter. Args arrive as one tuple.
        """

        self.assertFails(exc, needle, fn, *args, msg=msg)

    def assertSlots(self, fn, ok, cases, msg=None):
        """
        Every (slot, needle, bad values) case is refused, one slot at a time, the rest of
        ``ok`` left good.
        """
        for slot, needle, bads in cases:
            for bad in bads:
                arg = list(ok)
                arg[slot] = bad
                self.assertFails(
                    ValueError,
                    needle,
                    lambda a=arg: fn(*a),
                    msg=f"{msg} slot {slot} = {bad}",
                )
