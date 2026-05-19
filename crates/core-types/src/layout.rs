use crate::ids::DomainId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchLayout {
    pub batch_size: usize,
    pub sequence_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackedSequenceLayout {
    pub tokens: usize,
    pub sequences: usize,
    pub example_offsets: Vec<usize>,
    pub causal_start: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainPairLayout {
    pub source_domain: DomainId,
    pub target_domain: DomainId,
    pub pair_indices: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DomainTripletLayout {
    pub anchor_indices: Vec<usize>,
    pub positive_indices: Vec<usize>,
    pub negative_indices: Vec<usize>,
}
