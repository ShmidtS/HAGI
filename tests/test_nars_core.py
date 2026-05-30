import pytest

from hagi.nars import (
    GOAL,
    JUDGMENT,
    QUESTION,
    Atom,
    Bag,
    BudgetValue,
    Compound,
    Concept,
    Sentence,
    Task,
    Term,
    TruthValue,
    Var,
    budget_decay,
    merge_budgets,
    truth_abduction,
    truth_deduction,
    truth_induction,
    truth_intersection,
    truth_revision,
)


def test_term_atom_var_compound_creation_hash_and_equality():
    atom = Atom("bird")
    same_atom = Term.Atom("bird")
    var = Var("x")
    compound = Compound("inheritance", [atom, Atom("animal")])

    assert atom == same_atom
    assert hash(atom) == hash(same_atom)
    assert atom.is_atom
    assert var.name == "$x"
    assert var.is_var
    assert compound.is_compound
    assert compound.args == (atom, Atom("animal"))
    assert repr(compound) == "inheritance(bird, animal)"


def test_truth_value_operations_known_expected_values():
    t1 = TruthValue(0.8, 0.6)
    t2 = TruthValue(0.5, 0.4)

    revised = truth_revision(t1, t2)
    assert revised.frequency == pytest.approx(0.68)
    assert revised.confidence == pytest.approx(0.5)

    deduced = truth_deduction(t1, t2)
    assert deduced.frequency == pytest.approx(0.4)
    assert deduced.confidence == pytest.approx(0.24)

    induced = truth_induction(t1, t2)
    assert induced.frequency == pytest.approx(0.8)
    assert induced.confidence == pytest.approx(0.12)

    abducted = truth_abduction(t1, t2)
    assert abducted.frequency == pytest.approx(0.5)
    assert abducted.confidence == pytest.approx(0.192)

    intersected = truth_intersection(t1, t2)
    assert intersected.frequency == pytest.approx(0.4)
    assert intersected.confidence == pytest.approx(0.76)


def test_budget_merge_and_decay():
    left = BudgetValue(0.2, 0.7, 0.4)
    right = BudgetValue(0.9, 0.3, 0.8)

    assert merge_budgets(left, right) == BudgetValue(0.9, 0.7, 0.8)
    assert budget_decay(right, 0.5) == BudgetValue(0.45, 0.15, 0.8)
    assert BudgetValue(2.0, -1.0, 0.5) == BudgetValue(1.0, 0.0, 0.5)


def test_sentence_creation_with_supported_punctuation():
    term = Atom("goal")
    truth = TruthValue(0.9, 0.8)

    judgment = Sentence(term, JUDGMENT, truth=truth)
    question = Sentence(term, QUESTION)
    goal = Sentence(term, GOAL, truth=truth)

    assert judgment.punctuation == "."
    assert question.truth is None
    assert goal.punctuation == "!"

    with pytest.raises(ValueError):
        Sentence(term, QUESTION, truth=truth)
    with pytest.raises(ValueError):
        Sentence(term, JUDGMENT)


def test_task_executable_vs_question_detection():
    term = Atom("bird")
    truth = TruthValue(0.7, 0.6)
    budget = BudgetValue(0.5, 0.5, 0.5)

    judgment_task = Task(Sentence(term, JUDGMENT, truth=truth), budget)
    question_task = Task(Sentence(term, QUESTION), budget)

    assert judgment_task.is_executable()
    assert not judgment_task.is_question()
    assert not question_task.is_executable()
    assert question_task.is_question()


def test_concept_add_and_select_belief():
    term = Atom("bird")
    concept = Concept(term)
    weak = Sentence(term, JUDGMENT, truth=TruthValue(0.6, 0.4), stamp=1)
    strong = Sentence(term, JUDGMENT, truth=TruthValue(0.8, 0.9), stamp=2)
    desire = Sentence(term, GOAL, truth=TruthValue(0.7, 0.6))

    concept.add_belief(weak)
    concept.add_belief(strong)
    concept.add_desire(desire)

    assert concept.beliefs == [strong]
    assert concept.desires == [desire]
    assert concept.select_belief() is strong

    with pytest.raises(ValueError):
        concept.add_belief(Sentence(term, QUESTION))
    with pytest.raises(ValueError):
        concept.add_desire(strong)


def test_bag_put_take_get_deterministic_priority_ordering():
    low = Term.Atom("low")
    high_first = Term.Atom("high_first")
    high_second = Term.Atom("high_second")
    bag = Bag[Term]()

    bag.put(low, priority=0.2)
    bag.put(high_first, priority=0.9)
    bag.put(high_second, priority=0.9)

    assert len(bag) == 3
    assert bag.get("low") == low
    assert bag.priority("high_first") == pytest.approx(0.9)
    assert bag.take() == high_first
    assert bag.take() == high_second
    assert bag.take() == low
    assert bag.take() is None
