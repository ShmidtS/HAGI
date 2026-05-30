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

py-install:
    pip install -e .

py-test:
    pytest tests/ -v

py-lint:
    if command -v ruff >/dev/null 2>&1; then ruff check src/; else python -m compileall -q src/ scripts/; fi

py-train-smoke:
    hagi-train --config configs/overfit.yaml --device cpu --max-steps 10

py-eval-smoke:
    hagi-eval --checkpoint checkpoints/latest.pt --golden

py-lean-verify:
    python scripts/lean_verify.py
