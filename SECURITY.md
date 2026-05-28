# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in HAGI, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email: **security@hagi-project.dev** (or open a [private security advisory](https://github.com/ShmidtS/HAGI/security/advisories/new) on GitHub).
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix (if applicable)**: Within 30 days for critical issues

### Scope

This project is a research prototype. Security concerns most likely to be relevant:

- **Dependency vulnerabilities** in Python/Rust packages
- **Model serialization** — unsafe deserialization of checkpoint files
- **Data pipeline** — injection through training data or configs
- **CUDA kernels** — memory safety violations (future, Rust/CUDA stage)

### Out of Scope

- Model outputs (prompt injection, harmful content generation) — this is a research architecture, not a deployed service
- Performance-related issues
- Issues in third-party dependencies (report upstream)

## Security Best Practices for Contributors

- Never commit secrets, API keys, or credentials
- Use `pip audit` and `cargo audit` before submitting PRs
- Pin dependency versions in `requirements.txt` and `Cargo.lock`
- Review serialization code for unsafe deserialization patterns
