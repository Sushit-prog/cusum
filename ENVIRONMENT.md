# ENVIRONMENT.md — cusum-watch

Standing constraint, applies to every milestone. Read alongside INTERFACES.md,
TEST_TAXONOMY.md, VERSIONING_POLICY.md.

## Execution environment

All development, all shell commands, all installs, and all tests for this
project run inside **WSL2/Ubuntu 24.04**, never native Windows/PowerShell/CMD.

- Machine: GOKU — Intel i5-1235U, 8GB RAM, no GPU, Windows 11 host with WSL2.
- If a terminal panel defaults to PowerShell, switch it to a WSL/Ubuntu profile
  before running anything for this project. If you find yourself in
  PowerShell mid-task, stop and re-enter via `wsl` rather than continuing.
- Never install, invoke, or depend on Visual Studio, Visual Studio Build
  Tools, MSVC, or any native Windows compiler toolchain for this project.
  Every dependency must have a working install path via Linux prebuilt
  wheels/binaries inside WSL. If a `pip install` looks like it's about to
  compile from source, that's a signal to check for a Linux wheel mismatch
  (wrong Python version, wrong package name) — not a signal to reach for a
  Windows compiler.
- `llama-cpp-python` and any other native-extension Python package: install
  with `pip install <package> --break-system-packages` inside WSL. This
  should resolve to a prebuilt manylinux wheel with no compilation step. If
  it doesn't, stop and report which package/version needed to build from
  source, rather than installing a compiler to work around it.

## Why this exists

M1 nearly triggered a multi-GB Visual Studio Build Tools C++ workload install
on an 8GB-RAM machine because a shell defaulted to native PowerShell instead
of WSL, so `llama-cpp-python` had no prebuilt wheel to reach for and fell
back to source compilation. Caught and stopped before the workload actually
downloaded. This file exists so it doesn't recur on a later milestone.
