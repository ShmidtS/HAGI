use crate::dtype::DType;
use crate::shape::Shape;
use crate::tensor_layout::TensorLayout;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub shape: Shape,
    pub dtype: DType,
    pub layout: TensorLayout,
}

impl TensorSpec {
    pub fn new(shape: Shape, dtype: DType) -> Self {
        let layout = TensorLayout::contiguous(shape.clone(), dtype.size_of());
        Self {
            shape,
            dtype,
            layout,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}
