use std::fs::read;
use std::path::Path;
use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::{fftw_fftn, fftw_fftn_batched};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use lr_rs::rs_svd::{svd_hard, svd_soft};
use rayon::prelude::*;
use recon_lib::{estimate_phase_mask, grid_cartesian, grid_cartesian_f};



fn main() {

    let raw_dir = "/Users/Wyatt/l_plus_s_data";

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



fn __main() {

    // load each traj and raw, grid, perform SWT to estimate phase

    let raw_dir = "/Users/Wyatt/l_plus_s_data";

    let vol_shape = [512,256,256];
    let perm_shape = [256,256,512];
    let mask_shape = &vol_shape[1..];

    let vol_dims = ArrayDim::from_shape(&vol_shape);
    let perm_dims = ArrayDim::from_shape(&perm_shape);
    let mask_dims = ArrayDim::from_shape(&mask_shape);

    let n_vols = 150;
    for i in 0..n_vols {

        println!("working on vol {} of {} ...",i+1, n_vols);

        let raw = Path::new(raw_dir).join(format!("raw-{}",i));
        let trajf = Path::new(raw_dir).join(format!("traj-{}",i));
        let (data,dims) = read_cfl(&raw);
        let (traj,t_dims) = read_cfl(&trajf);
        let (mut x,mask) = grid_cartesian(&data,dims,&traj,t_dims,vol_dims,true);
        let m = mask.into_iter().map(|x|Complex32::new(x,0.)).collect::<Vec<_>>();


        // fft along x dimension
        fftw_fftn_batched(&mut x,&vol_shape[0..1],vol_shape[1..].iter().product(),FftDirection::Inverse,NormalizationType::Unitary);

        // fftshift the first dimension
        x.par_chunks_exact_mut(vol_shape[0]).for_each(|chunk| {
            let mut tmp = vec![Complex32::ZERO;chunk.len()];
            let a = ArrayDim::from_shape(&[vol_shape[0]]);
            a.fftshift(&chunk,&mut tmp,true);
            chunk.copy_from_slice(&tmp);
        });

        // permute to move x dim to back
        let mut tmp = perm_dims.alloc(Complex32::ZERO);
        tmp.par_iter_mut().enumerate().for_each(|(i,t)| {
            let [yi,zi,xi,..] = perm_dims.calc_idx(i);
            let addr = vol_dims.calc_addr(&[xi,yi,zi]);
            *t = x[addr];
        });

        // write y
        write_cfl(Path::new(raw_dir).join(format!("y-{}",i)),&tmp,perm_dims);
        write_cfl(Path::new(raw_dir).join(format!("m-{}",i)),&m,mask_dims);


    }



}


fn _main() {

    let base_dir = Path::new("/Users/Wyatt/l_plus_s_data");
    let raw_dims = ArrayDim::from_shape(&[512,256,256]);
    let permute_dims = ArrayDim::from_shape(&[256,256,11]);
    let prep_dims = ArrayDim::from_shape(&[256,256,11,16]);
    let mut prepped = prep_dims.alloc(Complex32::ZERO);

    let mask_dims = ArrayDim::from_shape(&[256,256,16]);
    let mut masks = mask_dims.alloc(0f32);
    let mut phase = prep_dims.alloc(Complex32::ONE);

    // prepped.chunks_exact_mut(permute_dims.numel())
    //     .zip(phase.chunks_exact_mut(permute_dims.numel()))
    //     .zip(masks.chunks_exact_mut(mask_dims.shape_squeeze()[0..2].iter().product()))
    //     .enumerate().for_each(|(vol_idx,((chunk,phase),mask))|{
    //
    //     println!("{}",vol_idx);
    //     let raw_name = format!("raw-{}",vol_idx + 1);
    //     let traj_name = format!("traj-{}",vol_idx + 1);
    //     let (data,dims) =read_cfl(base_dir.join(raw_name));
    //     let (traj,traj_dims) =read_cfl(base_dir.join(traj_name));
    //     let (mut g,d) = grid_cartesian(&data,dims,&traj,traj_dims,raw_dims,true);
    //
    //     mask.copy_from_slice(&d);
    //
    //     fftw_fftn_batched(&mut g,&raw_dims.shape()[0..1],raw_dims.shape()[1..].iter().product(),FftDirection::Inverse,NormalizationType::Unitary);
    //
    //     // fft shift the first readout dim
    //     g.par_chunks_exact_mut(raw_dims.shape()[0]).for_each(|readout|{
    //         let d = ArrayDim::from_shape(&raw_dims.shape()[0..1]);
    //         let src = readout.to_vec();
    //         d.fftshift(&src,readout,true);
    //     });
    //
    //
    //
    //     let mut buffer = permute_dims.alloc(Complex32::ZERO);
    //     // index and permute volume
    //     buffer.par_iter_mut().enumerate().for_each(|(s_idx,x)|{
    //         let [i,j,k,..] = permute_dims.calc_idx(s_idx);
    //         let addr = raw_dims.calc_addr(&[k+251,i,j]);
    //         *x = g[addr];
    //     });
    //     chunk.copy_from_slice(&buffer);
    //
    //     println!("estimating phase ...");
    //     let mut p = permute_dims.alloc(Complex32::ONE);
    //     fftw_fftn_batched(&mut buffer,&[256,256],11,FftDirection::Inverse,NormalizationType::Unitary);
    //     estimate_phase_mask(&buffer,&mut p,permute_dims,1.,6);
    //
    //     phase.copy_from_slice(&p);
    //     write_nifti("phase",&phase.iter().map(|x|x.to_polar().1).collect::<Vec<_>>(),permute_dims);
    //
    // });
    //
    // let masks:Vec<Complex32> = masks.into_iter().map(|x|Complex32::new(x,0.)).collect();
    //
    // write_nifti("phase",&phase.iter().map(|x|x.to_polar().1).collect::<Vec<_>>(),prep_dims);
    // write_cfl("prepped",&prepped,prep_dims);
    // write_cfl("masks",&masks,mask_dims);
    // write_cfl("phase",&phase,prep_dims);


    let (mut data,dims) = read_cfl("prepped");
    let (phase,dims) = read_cfl("phase");

    let llr_m = prep_dims.shape_squeeze()[0..3].iter().product();
    let llr_n = prep_dims.shape_squeeze()[3];
    let llr_dims = ArrayDim::from_shape(&[llr_m,llr_n]);

    assert_eq!(dims.numel(),llr_dims.numel());
    let mut dst = llr_dims.alloc(Complex32::ZERO);
    println!("running fft");
    fftw_fftn_batched(&mut data,&[256,256],11*16,FftDirection::Inverse,NormalizationType::Unitary);

    // do phase correction
    data.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
        *x *= p.conj();
    });

    // do shift

    svd_soft(&data,&mut dst,[llr_m,llr_n],2000.);

    // undo shift

    // undo phase correction
    dst.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
        *x *= p;
    });

    write_nifti("out",&dst.iter().map(|x|x.to_polar().0).collect::<Vec<_>>(),dims);

}