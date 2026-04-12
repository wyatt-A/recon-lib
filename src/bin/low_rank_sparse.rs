use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use ants_reg_wrapper::AntsRegistration;
use array_lib::ArrayDim;
use array_lib::cfl::num_traits::Zero;
use array_lib::io_cfl::{read_cfl, read_cfl_slice, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::{cu_fftn as fftn, cu_fftn_batch as fftn_batched};

#[cfg(not(feature = "cuda"))]
use dft_lib::fftw_fft::{fftw_fftn as fftn, fftw_fftn_batched as fftn_batched};




/// builds the main iterate for reconstruction (batch slices of each volume)
fn build_iterate(work_dir:impl AsRef<Path>, n:usize, batch_size:usize, slice_offset:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {


    let slice_stride = vol_dims[0] * vol_dims[1];

    let offset = slice_stride * slice_offset;
    let size = batch_size * slice_stride;

    let iterate_dims = ArrayDim::from_shape(&[vol_dims[0],vol_dims[1],batch_size,n]);
    let mut iterate = iterate_dims.alloc(Complex32::ZERO);

    let work_dir = work_dir.as_ref();
    iterate.par_chunks_exact_mut(size).enumerate().for_each(|(i,batch)|{
        let f = work_dir.join(format!("y-{}", i));
        read_cfl_slice(f,offset,batch);
    });

    (iterate, iterate_dims)

}

/// builds the phase mask for iterate
fn build_phase_map(work_dir:impl AsRef<Path>, n:usize, batch_size:usize, slice_offset:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {
    let slice_stride = vol_dims[0] * vol_dims[1];

    let offset = slice_stride * slice_offset;
    let size = batch_size * slice_stride;

    let phase_dims = ArrayDim::from_shape(&[vol_dims[0],vol_dims[1],batch_size,n]);
    let mut phase = phase_dims.alloc(Complex32::ONE);

    let work_dir = work_dir.as_ref();
    phase.par_chunks_exact_mut(size).enumerate().for_each(|(i,batch)|{
        let f = work_dir.join(format!("f-{}", i));
        read_cfl_slice(f,offset,batch);
        // normalize to unit magnitude
        batch.par_iter_mut().for_each(|x| *x = x.scale(1./x.norm()));
    });
    (phase, phase_dims)
}

fn apply_translations(work_dir:impl AsRef<Path>, n:usize, batch_size:usize, slice_offset:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {

    let mut f = File::open(work_dir.as_ref().join("trans.txt")).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let trans = s.lines().map(|l| {
        l.split_ascii_whitespace().map(|t| t.parse::<f32>().unwrap()).collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>();

    assert_eq!(trans.len(),n);

    // k-space coord determines the shift

    for i in 0..n {
        let f = work_dir.as_ref().join(format!("y-{}",i));
        let (data,dims) = read_cfl(&f);



    }




}

/// builds the sampling mask for iterate
fn build_sample_mask(work_dir:impl AsRef<Path>, n:usize, mask_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {

    let size = mask_dims[0] * mask_dims[1];
    let mask_dims = ArrayDim::from_shape(&[mask_dims[0],mask_dims[1],n]);
    let mut mask = mask_dims.alloc(Complex32::ZERO);

    let work_dir = work_dir.as_ref();
    mask.par_chunks_exact_mut(size).enumerate().for_each(|(i,mask)|{
        let f = work_dir.join(format!("m-{}", i));
        read_cfl_slice(f,0,mask);
    });

    (mask, mask_dims)

}



/// prepares input data for low-rank + sparse batch reconstruction
/// vol_dims must be the size of y. This is often the permuted volume shape where the fully-sampled
/// axis has been moved to the back i.e `[512,256,256]` -> `[256,256,512]`
fn prep_lrs(work_dir:impl AsRef<Path>, n:usize, batch_size:usize, batch_offset:usize, vol_dims:&[usize]) {
    let (it,it_dims) = build_iterate(&work_dir,n,batch_size,batch_offset,vol_dims);
    let (p,p_dims) = build_phase_map(&work_dir,n,batch_size,batch_offset,vol_dims);
    let (m,m_dims) = build_sample_mask(&work_dir,n,&vol_dims[0..2]);
    write_cfl(work_dir.as_ref().join("it"),&it,it_dims);
    write_cfl(work_dir.as_ref().join("p"),&p,p_dims);
    write_cfl(work_dir.as_ref().join("it_m"),&m,m_dims);
}




fn main() {

    let work_dir = "/Users/Wyatt/l_plus_s_data";

    prep_lrs(work_dir, 150, 11, 256, &[256,256,512]);

    let (raw,dims) = read_cfl(Path::new(work_dir).join("it"));
    let (phase,..) = read_cfl(Path::new(work_dir).join("p"));
    let (masks,..) = read_cfl(Path::new(work_dir).join("it_m"));


    let raw_shape = dims.shape_squeeze();
    let batch_size = raw_shape[2];
    let vols = raw_shape[3];

    // shape for 3-D wavelet xform
    let w_shape = &raw_shape[0..3];
    // 2-D fourier xform shape
    let f_shape = &raw_shape[0..2];
    // number of fourier transforms
    let n_f = raw_shape[2..4].iter().product();

    let batch_stride:usize = raw_shape[0..3].iter().product();
    let img_stride:usize = raw_shape[0..2].iter().product();

    let swt = SWT3Plan::new(w_shape, 5, Wavelet::new(WaveletType::Daubechies2));

    // forward model A
    let a = |x: &[Complex32], y: &mut [Complex32] | {
        y.copy_from_slice(x);
        // forward 2-D fft
        fftn_batched(y,f_shape,n_f,FftDirection::Forward,NormalizationType::Unitary);
        // apply sample mask
        y.chunks_exact_mut(batch_stride).zip(masks.chunks_exact(img_stride)).for_each(|(y,m)|{
            y.chunks_exact_mut(img_stride).for_each(|y|{
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
        y.chunks_exact_mut(batch_stride).zip(masks.chunks_exact(img_stride)).for_each(|(y,m)|{
            y.chunks_exact_mut(img_stride).for_each(|y|{
                y.iter_mut().zip(m).for_each(|(y,m)|{
                    *y *= m;
                })
            })
        });
        fftn_batched(y,f_shape,n_f,FftDirection::Forward,NormalizationType::Unitary);
    };

    let w = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(batch_stride).zip(x.chunks_exact(batch_stride)).for_each(|(y,x)|{
            swt.decompose(x,y);
        })
    };

    let wh = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(batch_stride).zip(x.chunks_exact(batch_stride)).for_each(|(y,x)|{
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