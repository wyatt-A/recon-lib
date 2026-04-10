use std::path::{Path, PathBuf};
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use clap::Parser;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::fftw_fftn_batched;
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    work_dir:PathBuf,
}


fn main() {


    let args = Args::parse();

    let raw_dir = &args.work_dir;

    // relative threshold value for SWT. This is proportional to sqrt(total energy)
    let rel_thresh = 1.5e-5;
    let decomp_levels = 4;

    let n = 150;

    for i in 0..n {

        println!("working on vol {} of {}",i+1,n);
        let (mut x,dims) = read_cfl(Path::new(raw_dir).join(format!("y-{}",i)));

        let vol_shape = dims.shape_squeeze();

        let swt = SWT3Plan::new(&vol_shape,decomp_levels,Wavelet::new(WaveletType::Daubechies2));
        let mut t = vec![Complex32::ZERO;swt.t_domain_size()];

        // fft first and second dims (y,z) [x was moved to back]
        fftw_fftn_batched(&mut x,&vol_shape[0..2],vol_shape[2],FftDirection::Inverse,NormalizationType::Unitary);

        // calculate total energy to determine threshold
        let te:f64 = x.par_iter().map(|x|x.norm_sqr() as f64).sum();

        println!("te: {}",te.sqrt());

        let lambda = te.sqrt() as f32 * rel_thresh;

        println!("decomposing ...");
        swt.decompose(&x,&mut t);
        println!("thresholding with lambda = {} ...",lambda);
        swt.soft_thresh(&mut t, lambda);
        println!("reconstructing ...");
        swt.reconstruct(&t, &mut x);
        write_cfl(Path::new(raw_dir).join(format!("f-{}",i)),&x,dims);
    }

}