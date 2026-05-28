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
}
