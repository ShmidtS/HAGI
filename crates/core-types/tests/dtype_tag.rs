use core_types::dtype::{DType, DTypeTag};

macro_rules! test_dtype_tag {
    ($name:ident, $rust_ty:ty, $variant:ident, $size:expr) => {
        #[test]
        fn $name() {
            assert_eq!(<$rust_ty as DTypeTag>::DTYPE, DType::$variant);
            assert_eq!(DType::$variant.size_of(), $size);
        }
    };
}

test_dtype_tag!(f32_dtype_tag, f32, F32, 4);
test_dtype_tag!(f64_dtype_tag, f64, F64, 8);
test_dtype_tag!(i32_dtype_tag, i32, I32, 4);
test_dtype_tag!(i64_dtype_tag, i64, I64, 8);
test_dtype_tag!(u8_dtype_tag, u8, U8, 1);
test_dtype_tag!(u32_dtype_tag, u32, U32, 4);
test_dtype_tag!(bool_dtype_tag, bool, Bool, 1);
