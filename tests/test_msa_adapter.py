import numpy as np

from hagi.msa import MSAAdapter
from hagi.nars import BudgetValue, JUDGMENT, Sentence, Task, Term, TruthValue


def _task(name: str, priority: float) -> Task:
    term = Term.Atom(name)
    sentence = Sentence(term, JUDGMENT, truth=TruthValue(0.8, 0.7))
    return Task(sentence, BudgetValue(priority, 0.5, 0.5))


def test_mask_shape_and_priority_columns():
    tasks = [_task("low", 0.1), _task("high", 0.9), _task("mid", 0.5)]
    adapter = MSAAdapter(top_k=2)

    mask = adapter.build_mask(tasks, seq_len=4)

    assert mask.shape == (4, 4)
    assert mask.dtype == bool or str(mask.dtype) == "torch.bool"
    assert mask[:, 1].all()
    assert mask[:, 2].all()
    assert not mask[:, 0].any()
    assert not mask[:, 3].any()


def test_route_kv_selects_priority_ordered_positions():
    tasks = [_task("low", 0.1), _task("high", 0.9), _task("mid", 0.5)]
    adapter = MSAAdapter(top_k=2)
    k = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    v = np.arange(12, 24, dtype=np.float32).reshape(1, 3, 4)

    routed_k, routed_v = adapter.route_kv(tasks, k, v)

    assert routed_k.shape == (1, 2, 4)
    assert routed_v.shape == (1, 2, 4)
    assert np.array_equal(routed_k, k[:, [1, 2], :])
    assert np.array_equal(routed_v, v[:, [1, 2], :])
