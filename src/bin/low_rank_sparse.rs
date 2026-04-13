use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use ants_reg_wrapper::AntsRegistration;
use array_lib::ArrayDim;
use array_lib::cfl::num_traits::Zero;
use array_lib::io_cfl::{read_cfl, read_cfl_slice, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use clap::Parser;
use conj_grad::CGSolver;
use dft_lib::common::{FftDirection, NormalizationType};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::{cu_fftn as fftn, cu_fftn_batch as fftn_batched};

#[cfg(not(feature = "cuda"))]
use dft_lib::fftw_fft::{fftw_fftn as fftn, fftw_fftn_batched as fftn_batched};
use lr_rs::rs_svd::svd_soft;


#[derive(Parser, Debug)]
struct Args {
    work_dir: PathBuf,
    index:usize,
}


fn main() {


    let args = Args::parse();
    let wd = &args.work_dir;
    let idx = args.index;

    let rho_w = 0.05;
    let lambda_w = 0.001;
    let rho_r = 0.05;
    let lambda_r = 0.005;
    let cg_tol = 1e-6;
    let cg_iter = 200;
    let admm_iter = 200;

    let (y,y_dims) = read_cfl(wd.join(format!("it-y-{}",idx)));
    let (phase,p_dims) = read_cfl(wd.join(format!("it-p-{}",idx)));
    let (m,m_dims) = read_cfl(wd.join(format!("it-m-{}",idx)));

    let y_shape = y_dims.shape_squeeze();
    let vols = y_shape[3];

    // shape for 3-D wavelet xform
    let w_shape = &y_shape[0..3];
    // 2-D fourier xform shape
    let f_shape = &y_shape[0..2];
    // number of fourier transforms
    let n_f = y_shape[2..4].iter().product();

    let batch_stride:usize = y_shape[0..3].iter().product();
    let img_stride:usize = y_shape[0..2].iter().product();

    let swt = SWT3Plan::new(w_shape, 5, Wavelet::new(WaveletType::Daubechies2));

    let mut x = vec![Complex32::ZERO;y_dims.numel()];
    let mut b = x.clone();
    let mut zw = vec![Complex32::ZERO;swt.t_domain_size()];
    let mut uw = zw.clone();
    let mut zr = x.clone();
    let mut ur = x.clone();

    let mut tmp_wx = zw.clone();
    let mut tmp_px = x.clone();

    // forward model A
    let a = |x: &[Complex32], y: &mut [Complex32] | {
        y.copy_from_slice(x);
        // forward 2-D fft
        fftn_batched(y,f_shape,n_f,FftDirection::Forward,NormalizationType::Unitary);
        // apply sample mask
        y.chunks_exact_mut(batch_stride).zip(m.chunks_exact(img_stride)).for_each(|(y,m)|{
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
        y.chunks_exact_mut(batch_stride).zip(m.chunks_exact(img_stride)).for_each(|(y,m)|{
            y.chunks_exact_mut(img_stride).for_each(|y|{
                y.iter_mut().zip(m).for_each(|(y,m)|{
                    *y *= m;
                })
            })
        });
        fftn_batched(y,f_shape,n_f,FftDirection::Inverse,NormalizationType::Unitary);
    };

    let w = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(swt.t_domain_size()).zip(x.chunks_exact(batch_stride)).for_each(|(y,x)|{
            swt.decompose(x,y);
        })
    };

    let wh = |x: &[Complex32], y:&mut [Complex32]| {
        y.chunks_exact_mut(batch_stride).zip(x.chunks_exact(swt.t_domain_size())).for_each(|(y,x)|{
            swt.reconstruct(x,y);
        })
    };

    // low-rank phase correction
    let p = |x: &[Complex32], y:&mut [Complex32]| {
        y.copy_from_slice(x);
        y.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
            *x *= p.conj();
        });
    };

    // inverse low-rank phase correction
    let ph = |x: &[Complex32], y:&mut [Complex32]| {
        y.copy_from_slice(x);
        y.iter_mut().zip(phase.iter()).for_each(|(x,p)|{
            *x *= p;
        });
    };

    // A^H y + rhow W^H (zw - uw) + rhor P^H (zr - ur)
    let rhs = |y:&[Complex32], zw:&[Complex32], zr:&[Complex32], uw:&[Complex32], ur:&[Complex32], b:&mut [Complex32]| {

        let mut tmp1 = vec![Complex32::ZERO; y.len()];
        let mut tmp2 = vec![Complex32::ZERO; zw.len()];
        let mut tmp3 = vec![Complex32::ZERO; y.len()];

        // tmp1 <- A^H y
        ah(y,&mut tmp1);

        // tmp2 <- (zw - uw)
        tmp2.iter_mut().zip(zw).zip(uw).for_each(|((t,z),u)|{
            *t = *z - *u;
        });

        println!("running inverse W");
        // b <- A^H y + rhow W^H (zw - uw)
        wh(&tmp2,b);
        println!("done");
        b.iter_mut().for_each(|b| *b = b.scale(rho_w));
        b.iter_mut().zip(&tmp1).for_each(|(a,b)| *a = *a + *b);

        // tmp3 <- (zr - ur)
        tmp3.iter_mut().zip(zr).zip(ur).for_each(|((t,z),u)|{
            *t = *z - *u;
        });

        // tmp1 <- rhor P^H (zr - ur)
        ph(&tmp3,&mut tmp1);
        tmp1.iter_mut().for_each(|t| *t = t.scale(rho_r));

        // b <- A^H y + rhow W^H (zw - uw) + rhor P^H (zr - ur)
        b.iter_mut().zip(&tmp1).for_each(|(a,b)| *a = *a + *b);

    };

    let prox_w = |wx:&[Complex32], uw:&[Complex32], zw:&mut [Complex32]| {
        zw.iter_mut().zip(uw).zip(wx).for_each(|((z,u),w)| *z = *z + *u + *w);
        swt.soft_thresh(zw,lambda_w/rho_w);
    };

    let prox_r = |px:&[Complex32], ur:&[Complex32], zr:&mut [Complex32]| {
        zr.iter_mut().zip(ur).zip(px).for_each(|((z,u),p)| *z = *z + *u + *p);
        let src = zr.to_vec();
        svd_soft(&src,zr,[batch_stride,vols],lambda_r/rho_r);
    };

    let dual_w = |wx:&[Complex32], zw:&[Complex32], uw:&mut [Complex32]| {
        uw.iter_mut().zip(wx).zip(zw).for_each(|((u,w),z)|{
            *u = *u + *w - *z;
        });
    };

    let dual_r = |px:&[Complex32], zr:&[Complex32], ur:&mut [Complex32]| {
        ur.iter_mut().zip(px).zip(zr).for_each(|((u,p),z)|{
            *u = *u + *p - *z;
        });
    };

    let lhs = |x: &[Complex32], y: &mut [Complex32]| {
        let mut tmp = vec![Complex32::ZERO; y.len()];

        // y = A^H A x
        a(x, &mut tmp);
        ah(&tmp, y);

        let rho = rho_w + rho_r;
        y.iter_mut().zip(x).for_each(|(yi, &xi)| {
            *yi += xi * rho;
        });
    };


    for it in 0..admm_iter {

        println!("it {}",it + 1);

        // x-update
        println!("updating linear system ...");
        rhs(&y,&zw, &zr, &uw, &ur, &mut b);
        println!("running CG solver ...");
        let mut cg = CGSolver::new(lhs);
        cg.report_residuals();
        cg.solve(&mut x,&b,cg_iter,cg_tol);

        println!("calculating split variables");
        w(&x,&mut tmp_wx);
        p(&x,&mut tmp_px);

        println!("running wavelet proximal update ...");
        prox_w(&tmp_wx, &uw, &mut zw);
        println!("running low-rank proximal update ...");
        prox_r(&tmp_px, &ur, &mut zr);

        println!("updating dual variables ...");
        dual_w(&tmp_wx, &zw, &mut uw);
        dual_r(&tmp_px, &zr, &mut ur);

        
    }

}
