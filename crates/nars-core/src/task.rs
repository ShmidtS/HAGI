use crate::{BudgetValue, Sentence};

#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    sentence: Sentence,
    budget: BudgetValue,
}

impl Task {
    pub fn new(sentence: Sentence, budget: BudgetValue) -> Self {
        Self { sentence, budget }
    }
    pub fn sentence(&self) -> &Sentence {
        &self.sentence
    }
    pub fn budget(&self) -> BudgetValue {
        self.budget
    }
    pub fn set_budget(&mut self, budget: BudgetValue) {
        self.budget = budget;
    }

    pub fn decay_budget(&mut self, factor: f64) {
        self.budget = self.budget.decay(factor);
    }

    pub fn merge_budget(&mut self, other: BudgetValue) {
        self.budget = self.budget.merge(other);
    }

    pub fn is_budget_valid(&self) -> bool {
        self.budget.is_valid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Term, TruthValue};

    #[test]
    fn new_stores_sentence_and_budget() {
        let sentence = Sentence::question(Term::atom("bird"));
        let budget = BudgetValue::new(0.4, 0.5, 0.6);
        let task = Task::new(sentence.clone(), budget);
        assert_eq!(task.sentence(), &sentence);
        assert_eq!(task.budget(), budget);
    }

    #[test]
    fn set_budget_replaces_budget_assignment() {
        let sentence = Sentence::judgment(Term::atom("bird"), TruthValue::new(1.0, 0.9), 1);
        let mut task = Task::new(sentence, BudgetValue::new(0.1, 0.2, 0.3));
        let budget = BudgetValue::new(0.7, 0.8, 0.9);
        task.set_budget(budget);
        assert_eq!(task.budget(), budget);
    }

    #[test]
    fn decay_budget_updates_inner_budget() {
        let sentence = Sentence::question(Term::atom("bird"));
        let mut task = Task::new(sentence, BudgetValue::new(0.8, 0.6, 0.4));
        task.decay_budget(0.5);
        assert_eq!(task.budget(), BudgetValue::new(0.4, 0.3, 0.4));
    }

    #[test]
    fn merge_budget_updates_inner_budget() {
        let sentence = Sentence::question(Term::atom("bird"));
        let mut task = Task::new(sentence, BudgetValue::new(0.2, 0.8, 0.4));
        task.merge_budget(BudgetValue::new(0.7, 0.3, 0.9));
        assert_eq!(task.budget(), BudgetValue::new(0.7, 0.8, 0.9));
        assert!(task.is_budget_valid());
    }
}
