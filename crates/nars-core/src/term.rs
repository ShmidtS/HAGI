#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Atom(String),
    Compound(String, Vec<Term>),
    Variable(String),
}

impl Term {
    pub fn atom(name: impl Into<String>) -> Self {
        Self::Atom(name.into())
    }
    pub fn compound(operator: impl Into<String>, terms: Vec<Term>) -> Self {
        Self::Compound(operator.into(), terms)
    }
    pub fn variable(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atom_constructs_named_atomic_term() {
        assert_eq!(Term::atom("bird"), Term::Atom("bird".to_string()));
    }

    #[test]
    fn compound_constructs_operator_with_subterms() {
        let term = Term::compound(
            "inheritance",
            vec![Term::atom("bird"), Term::atom("animal")],
        );
        assert_eq!(
            term,
            Term::Compound(
                "inheritance".to_string(),
                vec![
                    Term::Atom("bird".to_string()),
                    Term::Atom("animal".to_string())
                ]
            )
        );
    }

    #[test]
    fn variable_constructs_named_variable_term() {
        assert_eq!(Term::variable("x"), Term::Variable("x".to_string()));
    }
}
