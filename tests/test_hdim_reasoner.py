import numpy as np

from hagi.hdim import HDIMReasoner
from hagi.nars import Term


def test_encode_shape_and_determinism():
    reasoner = HDIMReasoner(dtype=np.float32)
    term = Term.Atom("bird")

    encoded_once = reasoner.encode(term)
    encoded_twice = reasoner.encode(term)

    assert encoded_once.shape == (8,)
    if hasattr(encoded_once, "detach"):
        assert bool((encoded_once == encoded_twice).all())
    else:
        assert np.array_equal(encoded_once, encoded_twice)


def test_reason_shape_and_determinism():
    reasoner = HDIMReasoner(dtype=np.float32)
    left = Term.Atom("bird")
    right = Term.Atom("animal")

    reasoned_once = reasoner.reason(left, right)
    reasoned_twice = reasoner.reason(left, right)

    assert reasoned_once.shape == (8,)
    if hasattr(reasoned_once, "detach"):
        assert bool((reasoned_once == reasoned_twice).all())
    else:
        assert np.array_equal(reasoned_once, reasoned_twice)
