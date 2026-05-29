use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/geometric_product.cu");
    println!("cargo:rerun-if-changed=kernels/rotor_sandwich.cu");
    println!("cargo:rerun-if-changed=kernels/sparse_attention.cu");
    println!("cargo:rerun-if-changed=kernels/hrm_update.cu");
    println!("cargo:rerun-if-changed=kernels/msa_route_score.cu");
    println!("cargo:rerun-if-changed=kernels/fused_rotor_hrm_msa.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    if let Some(cuda_lib_dir) = find_cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    let host_compiler = find_msvc_cl();

    for kernel in [
        "geometric_product",
        "rotor_sandwich",
        "sparse_attention",
        "hrm_update",
        "msa_route_score",
        "fused_rotor_hrm_msa",
    ] {
        let input = kernels_dir.join(format!("{kernel}.cu"));
        let output = out_dir.join(format!("{kernel}.ptx"));
        let mut command = Command::new("nvcc");
        command
            .arg("-ptx")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-arch=compute_70");
        if let Some(host_compiler) = &host_compiler {
            command.arg("-ccbin").arg(host_compiler);
        }
        let status = command
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .status()
            .unwrap_or_else(|err| panic!("failed to run nvcc for {}: {}", input.display(), err));

        if !status.success() {
            panic!("nvcc failed for {} with status {}", input.display(), status);
        }
    }
}

fn find_cuda_lib_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(path) = env::var_os(key) {
            candidates.push(PathBuf::from(path).join("lib").join("x64"));
        }
    }
    if let Some(program_files) = env::var_os("ProgramFiles") {
        candidates.push(
            PathBuf::from(program_files)
                .join("NVIDIA GPU Computing Toolkit")
                .join("CUDA"),
        );
    }

    for candidate in candidates {
        if candidate.join("cuda.lib").exists() {
            return Some(candidate);
        }
        let mut versions = std::fs::read_dir(&candidate)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path().join("lib").join("x64"))
            .filter(|path| path.join("cuda.lib").exists())
            .collect::<Vec<_>>();
        versions.sort();
        if let Some(path) = versions.pop() {
            return Some(path);
        }
    }
    None
}

fn find_msvc_cl() -> Option<PathBuf> {
    if let Some(path) = env::var_os("CUDAHOSTCXX").map(PathBuf::from) {
        return Some(path);
    }

    [
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.exists())
}
