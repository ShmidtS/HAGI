# cuda-kernels

CPU fallback kernels are available on stable Rust and are covered by parity tests.

Full CUDA backend requires nightly Rust + cargo oxide. See cuda-oxide documentation.

The `cuda` feature is scaffolding for host-side cuda-oxide integration and must remain optional so stable CPU builds continue to pass without compiling CUDA kernels.
