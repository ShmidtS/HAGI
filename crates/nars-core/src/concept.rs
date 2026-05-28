use crate::{Bag, Sentence, Task, Term, TruthValue};

#[derive(Debug, Clone, PartialEq)]
pub struct Concept {
    term: Term,
    beliefs: Bag<Task>,
    desires: Bag<Task>,
    questions: Bag<Task>,
}

impl Concept {
    pub fn new(term: Term) -> Self {
        Self {
            term,
            beliefs: Bag::new(),
            desires: Bag::new(),
            questions: Bag::new(),
        }
    }

    pub fn with_capacity(
        term: Term,
        belief_cap: usize,
        desire_cap: usize,
        question_cap: usize,
    ) -> Self {
        Self {
            term,
            beliefs: Bag::with_capacity_limit(belief_cap),
            desires: Bag::with_capacity_limit(desire_cap),
            questions: Bag::with_capacity_limit(question_cap),
        }
    }

    pub fn term(&self) -> &Term {
        &self.term
    }
    pub fn beliefs(&self) -> &Bag<Task> {
        &self.beliefs
    }
    pub fn desires(&self) -> &Bag<Task> {
        &self.desires
    }
    pub fn questions(&self) -> &Bag<Task> {
        &self.questions
    }
    pub fn accept(&mut self, task: Task) {
        match task.sentence() {
            Sentence::Judgment { term, truth, .. } => {
                self.accept_belief(task.clone(), term.clone(), truth.confidence())
            }
            Sentence::Goal { .. } => self.desires.put(task.clone(), task.budget().priority()),
            Sentence::Question { .. } => self.questions.put(task.clone(), task.budget().priority()),
        }
    }

    pub fn latest_belief_truth(&self) -> Option<&TruthValue> {
        self.beliefs
            .iter()
            .next()
            .and_then(|belief| match belief.sentence() {
                Sentence::Judgment { truth, .. } => Some(truth),
                _ => None,
            })
    }

    fn accept_belief(&mut self, task: Task, term: Term, confidence: f64) {
        let existing = self.beliefs.iter().any(|belief| {
            belief.sentence().term() == &term
                && belief
                    .sentence()
                    .truth()
                    .is_some_and(|truth| truth.confidence() >= confidence)
        });
        if !existing {
            self.beliefs
                .retain(|belief| belief.sentence().term() != &term);
            self.beliefs.put(task.clone(), task.budget().priority());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BudgetValue, TruthValue};

    fn task(sentence: Sentence, priority: f64) -> Task {
        Task::new(sentence, BudgetValue::new(priority, 0.5, 0.5))
    }

    #[test]
    fn new_starts_with_empty_beliefs_desires_and_questions() {
        let concept = Concept::new(Term::atom("bird"));
        assert_eq!(concept.term(), &Term::atom("bird"));
        assert!(concept.beliefs().is_empty());
        assert!(concept.desires().is_empty());
        assert!(concept.questions().is_empty());
    }

    #[test]
    fn accept_routes_judgments_to_beliefs() {
        let mut concept = Concept::new(Term::atom("bird"));
        concept.accept(task(
            Sentence::judgment(Term::atom("bird"), TruthValue::new(1.0, 0.7), 1),
            0.8,
        ));
        assert_eq!(concept.beliefs().len(), 1);
    }

    #[test]
    fn accept_routes_goals_to_desires() {
        let mut concept = Concept::new(Term::atom("food"));
        concept.accept(task(
            Sentence::goal(Term::atom("food"), TruthValue::new(1.0, 0.7), 1),
            0.8,
        ));
        assert_eq!(concept.desires().len(), 1);
    }

    #[test]
    fn accept_routes_questions_to_questions() {
        let mut concept = Concept::new(Term::atom("rain"));
        concept.accept(task(Sentence::question(Term::atom("rain")), 0.8));
        assert_eq!(concept.questions().len(), 1);
    }

    #[test]
    fn accept_replaces_lower_confidence_belief_for_same_term() {
        let mut concept = Concept::new(Term::atom("bird"));
        concept.accept(task(
            Sentence::judgment(Term::atom("bird"), TruthValue::new(0.4, 0.3), 1),
            0.2,
        ));
        concept.accept(task(
            Sentence::judgment(Term::atom("bird"), TruthValue::new(0.9, 0.8), 2),
            0.7,
        ));
        assert_eq!(concept.beliefs().len(), 1);
        let belief = concept.beliefs().iter().next().unwrap();
        assert_eq!(belief.sentence().truth().unwrap().confidence(), 0.8);
    }

    #[test]
    fn with_capacity_sets_bounded_bags_for_all_task_kinds() {
        let concept = Concept::with_capacity(Term::atom("bird"), 1, 2, 3);
        assert_eq!(concept.beliefs().capacity_limit(), Some(1));
        assert_eq!(concept.desires().capacity_limit(), Some(2));
        assert_eq!(concept.questions().capacity_limit(), Some(3));
    }

    #[test]
    fn accept_eviction_respects_configured_capacities() {
        let mut concept = Concept::with_capacity(Term::atom("bird"), 1, 1, 1);
        concept.accept(task(
            Sentence::judgment(Term::atom("sparrow"), TruthValue::new(0.4, 0.3), 1),
            0.2,
        ));
        concept.accept(task(
            Sentence::judgment(Term::atom("raven"), TruthValue::new(0.9, 0.8), 2),
            0.7,
        ));
        concept.accept(task(
            Sentence::goal(Term::atom("food"), TruthValue::new(1.0, 0.7), 1),
            0.2,
        ));
        concept.accept(task(
            Sentence::goal(Term::atom("water"), TruthValue::new(1.0, 0.7), 2),
            0.8,
        ));
        concept.accept(task(Sentence::question(Term::atom("rain")), 0.1));
        concept.accept(task(Sentence::question(Term::atom("snow")), 0.9));

        assert_eq!(concept.beliefs().len(), 1);
        assert_eq!(concept.desires().len(), 1);
        assert_eq!(concept.questions().len(), 1);
        assert_eq!(
            concept.beliefs().iter().next().unwrap().sentence().term(),
            &Term::atom("raven")
        );
        assert_eq!(
            concept.desires().iter().next().unwrap().sentence().term(),
            &Term::atom("water")
        );
        assert_eq!(
            concept.questions().iter().next().unwrap().sentence().term(),
            &Term::atom("snow")
        );
    }

    #[test]
    fn latest_belief_truth_returns_highest_priority_belief_truth() {
        let mut concept = Concept::new(Term::atom("bird"));
        let low_truth = TruthValue::new(0.2, 0.3);
        let high_truth = TruthValue::new(0.9, 0.8);
        concept.accept(task(
            Sentence::judgment(Term::atom("sparrow"), low_truth, 1),
            0.2,
        ));
        concept.accept(task(
            Sentence::judgment(Term::atom("raven"), high_truth, 2),
            0.7,
        ));

        assert_eq!(concept.latest_belief_truth(), Some(&high_truth));
    }
}
