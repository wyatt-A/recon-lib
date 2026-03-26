use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BartPicsSettings {
    /// "l1" or "l2"
    pub algo: String,
    pub lambda: f64,
    pub max_iter: usize,
    pub respect_scaling: bool,
    pub bin: PathBuf,
}

impl Default for BartPicsSettings {
    fn default() -> Self {
        Self {
            algo: "l1".to_string(),
            lambda: 0.005,
            max_iter: 50,
            respect_scaling: true,
            bin: PathBuf::from("bart"),
        }
    }
}

pub fn bart_pics(
    ksp: &[Complex32],
    dims: ArrayDim,
    settings: &BartPicsSettings,
    working_dir: impl AsRef<Path>,
) -> (Vec<Complex32>, ArrayDim) {
    if !working_dir.as_ref().exists() {
        create_dir_all(working_dir.as_ref()).expect("failed to create working dir");
    }

    let temp_ksp_in = working_dir.as_ref().join("temp_ksp_in");
    let temp_sens_in = working_dir.as_ref().join("temp_sens_in");
    let temp_img_out = working_dir.as_ref().join("temp_img_out");

    write_cfl(&temp_ksp_in,&ksp,dims);

    let sens = dims.alloc(Complex32::ONE);
    write_cfl(&temp_sens_in,&sens,dims);

    let mut cmd = Command::new(&settings.bin);
    let scale = if settings.respect_scaling { "-S" } else { "" };
    let debug = "-d5";
    cmd.arg("pics");
    cmd.arg(format!("-{}", settings.algo));
    cmd.arg(format!("-r{}", settings.lambda));
    cmd.arg(format!("-i{}", settings.max_iter));
    cmd.arg(scale);
    cmd.arg(debug);
    cmd.arg(&temp_ksp_in).arg(&temp_sens_in).arg(&temp_img_out);
    println!("{:?}", cmd);

    let c = cmd.stdin(Stdio::null());

    let child = c.spawn().expect("failed to spawn bart process");

    let o = child
        .wait_with_output()
        .expect("failed to wait for child process");

    if !o.status.success() {
        panic!("bart failed");
    }

    let (img,dims) = read_cfl(&temp_img_out);

    // do cleanup of temp files
    std::fs::remove_file(temp_img_out.with_extension("cfl")).expect("cannot clean temp image!");
    std::fs::remove_file(temp_img_out.with_extension("hdr"))
        .expect("cannot clean up temp image header!");

    std::fs::remove_file(temp_sens_in.with_extension("cfl")).expect("cannot clean sens image!");
    std::fs::remove_file(temp_sens_in.with_extension("hdr"))
        .expect("cannot clean up sense image header!");

    std::fs::remove_file(temp_ksp_in.with_extension("cfl")).expect("cannot clean kspace input!");
    std::fs::remove_file(temp_ksp_in.with_extension("hdr"))
        .expect("cannot clean up kspace input header!");

    (img,dims)
}