from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class TruthValue:
    frequency: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency", _clamp01(self.frequency))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))


def truth_revision(t1: TruthValue, t2: TruthValue) -> TruthValue:
    w1 = t1.confidence
    w2 = t2.confidence
    total_weight = w1 + w2
    if total_weight == 0.0:
        return TruthValue(0.5, 0.0)
    return TruthValue(
        (t1.frequency * w1 + t2.frequency * w2) / total_weight,
        total_weight / (1.0 + total_weight),
    )


def truth_deduction(t1: TruthValue, t2: TruthValue) -> TruthValue:
    return TruthValue(t1.frequency * t2.frequency, t1.confidence * t2.confidence)


def truth_induction(t1: TruthValue, t2: TruthValue) -> TruthValue:
    return TruthValue(t1.frequency, t1.confidence * t2.confidence * t2.frequency)


def truth_abduction(t1: TruthValue, t2: TruthValue) -> TruthValue:
    return TruthValue(t2.frequency, t1.confidence * t2.confidence * t1.frequency)


def truth_intersection(t1: TruthValue, t2: TruthValue) -> TruthValue:
    return TruthValue(
        t1.frequency * t2.frequency,
        t1.confidence + t2.confidence - t1.confidence * t2.confidence,
    )
