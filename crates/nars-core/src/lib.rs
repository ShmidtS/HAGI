pub mod bag;
pub mod budget;
pub mod concept;
pub mod sentence;
pub mod task;
pub mod term;
pub mod truth;

pub use bag::Bag;
pub use budget::BudgetValue;
pub use concept::Concept;
pub use sentence::Sentence;
pub use task::Task;
pub use term::Term;
pub use truth::TruthValue;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exports_public_api_types() {
        let term = Term::atom("bird");
        let truth = TruthValue::new(1.0, 0.9);
        let budget = BudgetValue::new(0.8, 0.7, 0.6);
        let task = Task::new(Sentence::judgment(term.clone(), truth, 1), budget);
        let concept = Concept::new(term);
        assert_eq!(task.budget(), budget);
        assert!(concept.beliefs().is_empty());
    }
}
