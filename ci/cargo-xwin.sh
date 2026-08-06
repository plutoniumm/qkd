#!/bin/sh
# CARGO shim for setuptools-rust when cross-compiling to windows-msvc: route
# compile commands through `cargo xwin` (which fetches the MSVC CRT + Windows
# SDK and links with lld); pass everything else (metadata, --version) to plain
# cargo, which cargo-xwin does not wrap.
#
# The conda env ships its own rust toolchain WITHOUT the windows-msvc std libs;
# rustup's toolchain (where ensure_xwin adds both windows targets) must win.
PATH="$HOME/.cargo/bin:$PATH"
export PATH
case "$1" in
  build | rustc | check) exec cargo xwin "$@" ;;
  *) exec cargo "$@" ;;
esac
