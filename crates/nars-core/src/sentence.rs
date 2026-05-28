use crate::{Term, TruthValue};

#[derive(Debug, Clone, PartialEq)]
pub enum Sentence {
    Judgment {
        term: Term,
        truth: TruthValue,
        stamp: u64,
    },
    Goal {
        term: Term,
        truth: TruthValue,
        stamp: u64,
    },
    Question {
        term: Term,
    },
}

impl Sentence {
    pub fn judgment(term: Term, truth: TruthValue, stamp: u64) -> Self {
        Self::Judgment { term, truth, stamp }
    }
    pub fn goal(term: Term, truth: TruthValue, stamp: u64) -> Self {
        Self::Goal { term, truth, stamp }
    }
    pub fn question(term: Term) -> Self {
        Self::Question { term }
    }
    pub fn term(&self) -> &Term {
        match self {
            Self::Judgment { term, .. } | Self::Goal { term, .. } | Self::Question { term } => term,
        }
    }
    pub fn truth(&self) -> Option<TruthValue> {
        match self {
            Self::Judgment { truth, .. } | Self::Goal { truth, .. } => Some(*truth),
            Self::Question { .. } => None,
        }
    }

    pub fn stamp(&self) -> Option<u64> {
        match self {
            Self::Judgment { stamp, .. } | Self::Goal { stamp, .. } => Some(*stamp),
            Self::Question { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judgment_stores_term_truth_and_stamp() {
        let term = Term::atom("bird");
        let truth = TruthValue::new(0.9, 0.8);
        let sentence = Sentence::judgment(term.clone(), truth, 42);
        assert_eq!(sentence.term(), &term);
        assert_eq!(sentence.truth(), Some(truth));
        assert_eq!(sentence.stamp(), Some(42));
    }

    #[test]
    fn goal_stores_term_truth_and_stamp() {
        let term = Term::atom("food");
        let truth = TruthValue::new(1.0, 0.7);
        let sentence = Sentence::goal(term.clone(), truth, 7);
        assert_eq!(sentence.term(), &term);
        assert_eq!(sentence.truth(), Some(truth));
        assert_eq!(sentence.stamp(), Some(7));
    }

    #[test]
    fn question_stores_term_without_truth_or_stamp() {
        let term = Term::atom("rain");
        let sentence = Sentence::question(term.clone());
        assert_eq!(sentence.term(), &term);
        assert_eq!(sentence.truth(), None);
        assert_eq!(sentence.stamp(), None);
    }
}
