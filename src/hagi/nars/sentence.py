from __future__ import annotations

from dataclasses import dataclass, field

from hagi.nars.budget import BudgetValue
from hagi.nars.term import Term
from hagi.nars.truth import TruthValue


JUDGMENT = "."
QUESTION = "?"
GOAL = "!"


@dataclass(frozen=True, slots=True)
class Sentence:
    term: Term
    punctuation: str
    truth: TruthValue | None = None
    stamp: int = 0
    budget: BudgetValue = field(default_factory=lambda: BudgetValue(0.5, 0.5, 0.5))

    def __post_init__(self) -> None:
        if self.punctuation not in {JUDGMENT, QUESTION, GOAL}:
            raise ValueError("punctuation must be one of '.', '?', '!'")
        if self.punctuation == QUESTION and self.truth is not None:
            raise ValueError("question sentences must not carry truth values")
        if self.punctuation in {JUDGMENT, GOAL} and self.truth is None:
            raise ValueError("judgment and goal sentences require truth values")
        if self.stamp < 0:
            raise ValueError("stamp must be non-negative")
