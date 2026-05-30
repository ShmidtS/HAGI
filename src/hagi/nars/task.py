from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from hagi.nars.budget import BudgetValue
from hagi.nars.sentence import GOAL, JUDGMENT, QUESTION, Sentence


@dataclass(slots=True)
class Task:
    sentence: Sentence
    budget: BudgetValue
    creation_time: int = 0
    best_solution: Self | None = None

    def is_executable(self) -> bool:
        return self.sentence.punctuation in {JUDGMENT, GOAL}

    def is_question(self) -> bool:
        return self.sentence.punctuation == QUESTION
