default:
    @just --list

test:
    cargo test --workspace

test-clifford:
    cargo test -p clifford-core

check:
    cargo check --workspace --all-targets

lint:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt --all

clean:
    cargo clean
