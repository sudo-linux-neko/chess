// Copyright (C) 2026 Vaibhav
// SPDX-License-Identifier: GPL-3.0-or-later

fn main() {
    // Enable AVX2 for x86_64 targets
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64" {
        println!("cargo:rustc-cfg=target_arch_x86_64");
        // Note: We use runtime feature detection (is_x86_feature_detected)
        // so we don't need compile-time target_feature
    }
    
    // Enable NEON for aarch64 targets
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "aarch64" {
        println!("cargo:rustc-cfg=target_arch_aarch64");
    }

    pyo3_build_config::add_extension_module_link_args();
}