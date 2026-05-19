# WSL2 Setup Guide for HAGI / cuda-oxide

cuda-oxide is Linux-only. On Windows, use WSL2 with a Linux distribution.

## Prerequisites

- Windows 11 (recommended) or Windows 10 2004+
- Administrator access in PowerShell
- NVIDIA GPU with driver supporting CUDA on WSL2

## Step 1: Install WSL2 + Ubuntu 24.04

Open **PowerShell as Administrator** and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\setup\wsl2-install.ps1
```

This installs:
- WSL2 kernel
- Virtual Machine Platform
- Ubuntu 24.04 distribution

**Reboot when prompted.**

After reboot, launch Ubuntu:
```powershell
wsl -d Ubuntu-24.04
```

Complete the initial user setup (username + password).

## Step 2: Install Dependencies Inside WSL2

From within the Ubuntu WSL2 shell:

```bash
bash /mnt/e/HAGI/setup/ubuntu-setup.sh
```

This script installs:

| Component | Version | Purpose |
|---|---|---|
| CUDA Toolkit | 12.8 | nvcc, CUDA runtime, libdevice |
| LLVM | 21 | PTX code generation via NVPTX backend |
| Clang | 21 | bindgen host bindings |
| Rust (nightly) | pinned | cuda-oxide requires nightly |
| cargo-oxide | latest | cargo subcommand for cuda-oxide builds |

**Verification after install:**

```bash
nvcc --version              # CUDA compiler
llc-21 --version | grep nvptx  # NVPTX backend present
cargo --version             # Rust package manager
cargo oxide --version       # cuda-oxide tool (from cuda-oxide repo)
```

## Step 3: Clone cuda-oxide Repository

cuda-oxide uses a pinned Rust nightly via `rust-toolchain.toml`. You typically build inside its repo:

```bash
git clone https://github.com/NVlabs/cuda-oxide ~/cuda-oxide
cd ~/cuda-oxide
cargo oxide doctor          # Verify all prerequisites
cargo oxide run vecadd      # Test with sample kernel
```

## Step 4: Build HAGI

```bash
cd /mnt/e/HAGI
cargo check --workspace     # Standard Rust crates (CPU reference)
```

cuda-oxide kernel compilation will be added to `crates/cuda-kernels` once the kernel API stabilizes.

## CUDA on WSL2 Notes

- Host NVIDIA driver must be installed on Windows (not inside WSL).
- CUDA Toolkit is installed **inside** WSL2.
- `nvidia-smi` works from both Windows and WSL2 if drivers are compatible.
- GPU compute is shared between Windows host and WSL2.

## Troubleshooting

### "NVPTX backend not found"

LLVM must be built with `NVPTX` target. The `llvm.sh` script from apt.llvm.org includes it by default. If you build from source, pass:

```
-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"
```

### "cargo oxide: command not found"

`cargo-oxide` must be installed from the cuda-oxide repo:

```bash
cd ~/cuda-oxide
cargo install --path crates/cargo-oxide
```

### "Windows is not supported"

cuda-oxide requires a Linux environment. Always run `cargo oxide` commands inside WSL2, never from Windows PowerShell/CMD.

## References

- [cuda-oxide Installation](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)
- [CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVlabs/cuda-oxide GitHub](https://github.com/NVlabs/cuda-oxide)
