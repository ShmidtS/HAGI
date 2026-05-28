//! FFI compatibility layer.
//!
//! C ABI bindings are intentionally deferred; this crate currently exposes
//! Rust-side compatibility metadata for downstream bridge crates.

/// Current FFI compatibility API version.
pub const API_VERSION: &str = "0.1.0";

/// Returns the FFI compatibility API version string.
pub fn version() -> &'static str {
    API_VERSION
}

/// Returns the FFI compatibility API version string.
pub fn ffi_compat_version() -> &'static str {
    version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_stub() {
        assert_ne!(version(), "0.1.0-stub");
        assert_eq!(version(), API_VERSION);
    }
}
