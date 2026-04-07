use array_lib::ArrayDim;
use array_lib::io_cfl::read_cfl;
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use recon_lib::filters::Fermi;
use recon_lib::grid_cartesian_f;
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::{cu_fftn as fftn, cu_fftn_batch as fftn_batched};

#[cfg(not(feature = "cuda"))]
use dft_lib::fftw_fft::{fftw_fftn as fftn, fftw_fftn_batched as fftn_batched};


fn main() {

    let lambda = 0.0002;
    let rho = 0.01;
    let n_it = 300;

    {
        let (raw,raw_dims) = read_cfl(r"/Users/Wyatt/l_plus_s_data/out-0.cfl");
        write_nifti("ref",&raw.iter().map(|x|x.norm()).collect::<Vec<_>>(),raw_dims);
    }

    let (raw,raw_dims) = read_cfl(r"/Users/Wyatt/l_plus_s_data/raw-0.cfl");
    let (traj,traj_dims) = read_cfl(r"/Users/Wyatt/l_plus_s_data/traj-0.cfl");

    let grid_dims = ArrayDim::from_shape(&[512,256,256]);
    let grid_shape = grid_dims.shape_ns();

    let (mut g,mask) = {
        let (mut g,mask) = grid_cartesian_f::<Fermi>(&raw, raw_dims, &traj, traj_dims, grid_dims, true, None);
        g.iter_mut().for_each(|x| *x = x.scale(1./28.));
        // let mut shifted = grid_dims.alloc(Complex32::ZERO);
        // grid_dims.fftshift(&g,&mut shifted,true);
        (g,mask)
    };

    // 1-D fft along x dimension
    fftn_batched(&mut g,&[grid_shape[0]],grid_shape[1]*grid_shape[2],FftDirection::Inverse,NormalizationType::Unitary);

    let permute_shape = [grid_shape[1],grid_shape[2],grid_shape[0]];
    let permute_dims = ArrayDim::from_shape(permute_shape.as_slice());
    {
        let mut permuted = permute_dims.alloc(Complex32::ZERO);
        permuted.par_iter_mut().enumerate().for_each(|(i,s)|{
            let [y,z,x,..] = permute_dims.calc_idx(i);
            let addr = grid_dims.calc_addr(&[x,y,z]);
            *s = g[addr];
        });
        g = permuted;
    }

    let permute_shape = [grid_shape[1],grid_shape[2],10];
    let permute_dims = ArrayDim::from_shape(permute_shape.as_slice());
    let mut g = g[0..permute_dims.numel()].to_owned();

    // k-space for data consistency
    let y = g.clone();

    fftn_batched(&mut g,&permute_shape[0..2],permute_shape[2],FftDirection::Inverse,NormalizationType::Unitary);

    let mut x = g;
    write_nifti("x.nii",&x.iter().map(|x|x.norm()).collect::<Vec<_>>(),permute_dims);

    let swt = SWT3Plan::new(&permute_shape,5,Wavelet::new(WaveletType::Daubechies2));

    // forward model A
    let a = |x: &[Complex32], y: &mut [Complex32] | {
        y.copy_from_slice(x);
        fftn_batched(y,&permute_shape[0..2],permute_shape[2],FftDirection::Forward,NormalizationType::Unitary);
        y.chunks_exact_mut(permute_shape[0]*permute_shape[1]).zip(&mask).for_each(|(y,m)|{
            y.iter_mut().zip(&mask).for_each(|(y,m)|{
                *y *= m;
            })
        });
    };
    //
    // // inverse model A^H
    let ah = |x: &[Complex32], y: &mut [Complex32]| {
        y.copy_from_slice(x);
        y.chunks_exact_mut(permute_shape[0]*permute_shape[1]).for_each(|y|{
            y.iter_mut().zip(&mask).for_each(|(y,m)|{
                *y *= m;
            })
        });
        fftn_batched(y,&permute_shape[0..2],permute_shape[2],FftDirection::Inverse,NormalizationType::Unitary);
    };

    let w = |x: &[Complex32], y:&mut [Complex32]| {
        swt.decompose(x,y);
    };

    let wh = |x: &[Complex32], y:&mut [Complex32]| {
        swt.reconstruct(x,y);
    };
    //
    // z0 = W x0
    let mut z = vec![Complex32::ZERO; swt.t_domain_size()];
    //w(&x,&mut z);

    // u0 = 0;
    let mut u = vec![Complex32::ZERO; swt.t_domain_size()];

    // b = A^H y + rho W^H (z - u)
    let calc_b = |y_meas: &[Complex32], z: &[Complex32], u: &[Complex32], b: &mut [Complex32]| {
        ah(y_meas, b);

        let mut tmp = z.to_vec();
        tmp.iter_mut().zip(u).for_each(|(t, &ui)| {
            *t -= ui;
        });

        let mut tmp_img = vec![Complex32::ZERO; b.len()];
        wh(&tmp, &mut tmp_img);

        b.iter_mut().zip(&tmp_img).for_each(|(bi, &ti)| {
            *bi += rho * ti;
        });
    };
    //
    // x-update operator: y = A^H A x + rho W^H W x
    let cga = |x: &[Complex32], y: &mut [Complex32]| {
        let mut tmp_k = vec![Complex32::ZERO; x.len()];
        let mut tmp_img = vec![Complex32::ZERO; x.len()];
        a(x, &mut tmp_k);
        ah(&tmp_k, y);
        tmp_img.copy_from_slice(x);
        y.iter_mut().zip(&tmp_img).for_each(|(yi, &wi)| {
            *yi += rho * wi;
        });
    };

    let mut b = permute_dims.alloc(Complex32::ZERO);
    let mut xtmp = b.clone();

    for it in 0..n_it {

        println!("it {}",it);
        // update x

        calc_b(&y, &z, &u, &mut b);
        let mut cg = conj_grad::CGSolver::new(&cga);
        cg.report_residuals();
        //cg.report_residuals();
        cg.solve(&mut xtmp, &b, 200, 1e-6);
        x.copy_from_slice(&xtmp);

        // calculate data consistency
        a(xtmp.as_slice(), &mut b);
        let dc = y.iter().zip(b.iter()).map(|(a,b)| (a - b).norm_sqr()).sum::<f32>();
        println!("dc = {:.6e}", dc / b.len() as f32);

        // update z
        let mut wtmp = vec![Complex32::ZERO; swt.t_domain_size()];
        w(&x, &mut wtmp);
        wtmp.iter_mut().zip(&u).for_each(|(w, u)| {
            *w += u;
        });

        swt.soft_thresh(&mut wtmp, lambda / rho);
        let z_prev = z.clone();
        z.copy_from_slice(&wtmp);

        let l1 = wtmp.iter().map(|x|x.norm()).sum::<f32>();
        println!("l1 = {:.6e}", l1 / b.len() as f32);

        // update u
        w(&x, &mut wtmp);
        wtmp.iter_mut().zip(u.iter().zip(&z)).for_each(|(w,(u,z))|{
            *w += (u - z);
        });
        u.copy_from_slice(&wtmp);

        let mut wx = vec![Complex32::ZERO; swt.t_domain_size()];
        w(&x, &mut wx);

        // primal residual
        let r_norm = wx.iter()
            .zip(&z)
            .map(|(&a, &b)| (a - b).norm_sqr())
            .sum::<f32>()
            .sqrt();

        let wx_norm = wx.iter().map(|v| v.norm_sqr()).sum::<f32>().sqrt();
        let z_norm  = z.iter().map(|v| v.norm_sqr()).sum::<f32>().sqrt();

        let eps_abs = 1e-5f32;
        let eps_rel = 1e-5f32;

        let eps_pri = (z.len() as f32).sqrt() * eps_abs
            + eps_rel * wx_norm.max(z_norm);

        // dual residual
        let mut dz = z.clone();
        dz.iter_mut().zip(&z_prev).for_each(|(d, &zp)| {
            *d -= zp;
        });

        let mut wh_dz = vec![Complex32::ZERO; x.len()];
        wh(&dz, &mut wh_dz);

        let s_norm = wh_dz.iter()
            .map(|&v| (rho * v).norm_sqr())
            .sum::<f32>()
            .sqrt();

        // one common form
        let u_img_norm = {
            let mut wh_u = vec![Complex32::ZERO; x.len()];
            wh(&u, &mut wh_u);
            wh_u.iter().map(|v| (rho * *v).norm_sqr()).sum::<f32>().sqrt()
        };

        let eps_dual = (x.len() as f32).sqrt() * eps_abs
            + eps_rel * u_img_norm;

        if (s_norm < eps_dual) {
            break
        }
        
        if (r_norm < eps_pri) &&  (s_norm < eps_dual) {
            break
        }

        println!(
            "r = {:.6e}, eps_pri = {:.6e}, s = {:.6e}, eps_dual = {:.6e}",
            r_norm, eps_pri, s_norm, eps_dual
        );

        let out = x.iter().map(|x| x.norm()).collect::<Vec<_>>();
        let mut shifted = out.clone();
        permute_dims.fftshift(&out,&mut shifted,true);
        write_nifti("out",&shifted,permute_dims);

    }

    let out = x.iter().map(|x| x.norm()).collect::<Vec<_>>();
    let mut shifted = out.clone();
    permute_dims.fftshift(&out,&mut shifted,true);
    write_nifti("out",&shifted,permute_dims);


}