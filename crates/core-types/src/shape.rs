#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn checked_numel(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }
}
