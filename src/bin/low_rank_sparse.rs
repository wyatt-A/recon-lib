use array_lib::ArrayDim;
use array_lib::io_cfl::read_cfl;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::{cu_fftn as fftn, cu_fftn_batch as fftn_batched};

#[cfg(not(feature = "cuda"))]
use dft_lib::fftw_fft::{fftw_fftn as fftn, fftw_fftn_batched as fftn_batched};

fn main() {

    let (raw,dims) = read_cfl("prepped");
    let (phase,..) = read_cfl("phase");
    let (masks,..) = read_cfl("masks");

    let ds = dims.shape();
    let w3_dims = ArrayDim::from_shape(&[ds[0],ds[1],ds[2]]);
    let f2_dims = ArrayDim::from_shape(&[ds[0],ds[1]]);
    let f_batches = ds[2] * ds[3];


    let swt = SWT3Plan::new(w3_dims.shape(), 5, Wavelet::new(WaveletType::Daubechies2));

    // forward model A
    let a = |x: &[Complex32], y: &mut [Complex32] | {
        y.copy_from_slice(x);
        fftn_batched(y,&f2_dims.shape_squeeze(),f_batches,FftDirection::Forward,NormalizationType::Unitary);
        y.chunks_exact_mut(f2_dims.numel() * ds[2]).zip(masks.chunks_exact(f2_dims.numel())).for_each(|(y,m)|{
            y.chunks_exact_mut(f2_dims.numel()).for_each(|y|{
                y.iter_mut().zip(m).for_each(|(y,m)|{
                    *y *= m;
                })
            })
        });
    };
    //
    // // inverse model A^H
    let ah = |x: &[Complex32], y: &mut [Complex32]| {
        y.copy_from_slice(x);
        y.chunks_exact_mut(f2_dims.numel() * ds[2]).zip(masks.chunks_exact(f2_dims.numel())).for_each(|(y,m)|{
            y.chunks_exact_mut(f2_dims.numel()).for_each(|y|{
                y.iter_mut().zip(m).for_each(|(y,m)|{
                    *y *= m;
                })
            })
        });
        fftn_batched(y,&f2_dims.shape_squeeze(),f_batches,FftDirection::Forward,NormalizationType::Unitary);
    };

    let w = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(w3_dims.numel()).zip(x.chunks_exact(w3_dims.numel())).for_each(|(y,x)|{
            swt.decompose(x,y);
        })
    };

    let wh = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(w3_dims.numel()).zip(x.chunks_exact(w3_dims.numel())).for_each(|(y,x)|{
            swt.reconstruct(x,y);
        })
    };

    // low-rank phase correction
    let l = |x: &[Complex32], y:&mut [Complex32]| {
        y.copy_from_slice(x);
        y.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
            *x *= p.conj();
        });
    };

    // inverse low-rank phase correction
    let lh = |x: &[Complex32], y:&mut [Complex32]| {
        y.copy_from_slice(x);
        y.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
            *x *= p;
        });
    };




}