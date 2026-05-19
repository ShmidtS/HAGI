#!/bin/bash
# Setup script for HAGI dependencies inside WSL2 Ubuntu 24.04
# Run inside WSL: bash /mnt/e/HAGI/setup/ubuntu-setup.sh

set -euo pipefail

echo "=== HAGI WSL2 Setup ==="
echo "This installs all dependencies for cuda-oxide development"

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install base tools
sudo apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    lsb-release \
    software-properties-common \
    gnupg \
    cmake \
    ninja-build

# === CUDA Toolkit ===
echo "=== Installing CUDA Toolkit ==="
# CUDA Toolkit for WSL2 (follows same path as Linux)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
rm -f cuda-keyring_1.1-1_all.deb

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH

# === LLVM 21+ with NVPTX backend ===
echo "=== Installing LLVM 21 ==="
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21
rm -f llvm.sh

sudo apt-get install -y \
    llvm-21 \
    llvm-21-dev \
    clang-21 \
    libclang-21-dev

# Verify NVPTX backend
if ! llc-21 --version | grep -q nvptx; then
    echo "ERROR: NVPTX backend not found in LLVM 21" >&2
    echo "cuda-oxide requires LLVM built with NVPTX target" >&2
    exit 1
fi
echo "NVPTX backend confirmed in LLVM 21"

# === Rust (nightly for cuda-oxide) ===
echo "=== Installing Rust via rustup ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Pin nightly for cuda-oxide
echo "=== Installing Rust nightly (required by cuda-oxide) ==="
rustup install nightly-2026-04-15
rustup default nightly-2026-04-15
rustup component add rust-src

# === Install cuda-oxide ===
echo "=== Installing cuda-oxide ==="
cargo install --git https://github.com/NVlabs/cuda-oxide cargo-oxide

# === Clone HAGI repo for development ===
echo "=== Setup complete ==="
echo ""
echo "CUDA Toolkit: $(nvcc --version | head -1)"
echo "LLVM: $(llc-21 --version | head -1)"
echo "Rust: $(rustc --version)"
echo "cargo-oxide: $(cargo oxide --version 2>/dev/null || echo 'cargo-oxide: run inside cuda-oxide repo')"
echo ""
echo "To build HAGI:"
echo "  cd /mnt/e/HAGI"
echo "  cargo check --workspace"
echo ""
echo "For cuda-oxide kernel compilation, clone the cuda-oxide repo:"
echo "  git clone https://github.com/NVlabs/cuda-oxide ~/cuda-oxide"
echo "  cd ~/cuda-oxide"
echo "  cargo oxide doctor"
