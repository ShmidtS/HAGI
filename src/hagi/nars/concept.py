from __future__ import annotations

from dataclasses import dataclass, field

from hagi.nars.budget import BudgetValue
from hagi.nars.sentence import GOAL, JUDGMENT, QUESTION, Sentence
from hagi.nars.task import Task
from hagi.nars.term import Term


@dataclass(slots=True)
class Concept:
    term: Term
    beliefs: list[Sentence] = field(default_factory=list)
    desires: list[Sentence] = field(default_factory=list)
    questions: list[Sentence] = field(default_factory=list)
    budget: BudgetValue = field(default_factory=lambda: BudgetValue(0.5, 0.5, 0.5))
    tasks: list[Task] = field(default_factory=list)

    def add_belief(self, sentence: Sentence) -> None:
        if sentence.punctuation != JUDGMENT:
            raise ValueError("belief must be a judgment sentence")
        existing_index = next(
            (index for index, belief in enumerate(self.beliefs) if belief.term == sentence.term),
            None,
        )
        if existing_index is None:
            self.beliefs.append(sentence)
            return
        existing = self.beliefs[existing_index]
        existing_confidence = existing.truth.confidence if existing.truth is not None else 0.0
        new_confidence = sentence.truth.confidence if sentence.truth is not None else 0.0
        if new_confidence >= existing_confidence:
            self.beliefs[existing_index] = sentence

    def add_desire(self, sentence: Sentence) -> None:
        if sentence.punctuation != GOAL:
            raise ValueError("desire must be a goal sentence")
        self.desires.append(sentence)

    def add_question(self, sentence: Sentence) -> None:
        if sentence.punctuation != QUESTION:
            raise ValueError("question must be a question sentence")
        self.questions.append(sentence)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)
        if task.sentence.punctuation == JUDGMENT:
            self.add_belief(task.sentence)
        elif task.sentence.punctuation == GOAL:
            self.add_desire(task.sentence)
        elif task.sentence.punctuation == QUESTION:
            self.add_question(task.sentence)

    def select_belief(self) -> Sentence | None:
        if not self.beliefs:
            return None
        return max(
            self.beliefs,
            key=lambda belief: (
                belief.truth.confidence if belief.truth is not None else 0.0,
                belief.budget.priority,
                -belief.stamp,
                repr(belief.term),
            ),
        )
