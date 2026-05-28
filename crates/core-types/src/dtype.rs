#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
    U32,
    Bool,
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U32 => 4,
            DType::Bool => 1,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait DTypeTag: Copy + Default + sealed::Sealed + Send + Sync + 'static {
    const DTYPE: DType;
}

macro_rules! impl_dtype_tag {
    ($rust_ty:ty, $variant:ident) => {
        impl sealed::Sealed for $rust_ty {}
        impl DTypeTag for $rust_ty {
            const DTYPE: DType = DType::$variant;
        }
    };
}

impl_dtype_tag!(f32, F32);
impl_dtype_tag!(f64, F64);
impl_dtype_tag!(i32, I32);
impl_dtype_tag!(i64, I64);
impl_dtype_tag!(u8, U8);
impl_dtype_tag!(u32, U32);
impl_dtype_tag!(bool, Bool);
