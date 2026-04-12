use std::f64::consts::PI;
use std::fs::{create_dir_all, read};
use std::path::{Path, PathBuf};
use ants_reg_wrapper::AntsRegistration;
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

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    sub_cmd: SubCmd,
}

#[derive(Args, Debug, Clone)]
struct FilterArgs {
    work_dir:PathBuf,
    index:usize,
    x:usize,
    y:usize,
    z:usize,
}

#[derive(Args, Debug, Clone)]
struct GenYArgs {
    work_dir:PathBuf,
    index:usize,
    ref_index:usize,
    x:usize,
    y:usize,
    z:usize,
}

#[derive(Args, Debug, Clone)]
struct PhaseArgs {
    work_dir:PathBuf,
    index:usize,
    x:usize,
    y:usize,
    z:usize,
}


#[derive(Subcommand, Debug, Clone)]
pub enum SubCmd {
    /// Run reconstruction
    Filter(FilterArgs),
    GenY(GenYArgs),
    GenPhase(PhaseArgs),
}

fn main() {
    let args = CliArgs::parse();
    match args.sub_cmd {
        SubCmd::Filter(args) => {
            filter_images(args.work_dir,args.index,&[args.x,args.y,args.z]);
        }
        SubCmd::GenY(args) => {
            generate_y(args.work_dir,args.index,args.ref_index,&[args.x,args.y,args.z])
        }
        SubCmd::GenPhase(args) => {
            generate_phase_maps(args.work_dir,args.index,&[args.x,args.y,args.z])
        }
    }
}


/// corrects the measurement data by applying a linear phase ramp to k-space for a first-order
/// eddy current shift correction. We commonly see a large shift along the frequency encoding axis.
/// This is meant to reduce the rank of the data set over q-space
///
///
/// Generates a set of filtered images by zero-filling and applying WST. This is used for estimating
/// smooth phase maps and translations for reducing the q-space rank.
fn filter_images(work_dir:impl AsRef<Path>, i:usize, vol_dims:&[usize]) {

    let wd = work_dir.as_ref();

    let out_dir = wd.join("filtered");
    create_dir_all(&out_dir).unwrap();

    let cfl_out = out_dir.join(format!("f-{}",i));
    let raw_file = wd.join(format!("raw-{}",i));
    let traj_file = wd.join(format!("traj-{}",i));

    let rel_thresh = 1.5e-5;
    let decomp_levels = 4;

    let swt = SWT3Plan::new(vol_dims,decomp_levels,Wavelet::new(WaveletType::Daubechies2));

    let vol_d = ArrayDim::from_shape(vol_dims);

    let mut tmp_buff = vol_d.alloc(Complex32::ZERO);
    let mut t_dom = vec![Complex32::ZERO;swt.t_domain_size()];

    let (raw,raw_dims) = read_cfl(&raw_file);
    let (traj,traj_dims) = read_cfl(&traj_file);

    let (mut g,..) = grid_cartesian(&raw,raw_dims,&traj,traj_dims,vol_d,true);

    // inverse 3-D fft
    fftw_fftn(&mut g,vol_dims,FftDirection::Inverse,NormalizationType::Unitary);
    vol_d.fftshift(&g, &mut tmp_buff, true);

    let total_energy:f64 = tmp_buff.par_iter().map(|x|x.norm_sqr() as f64).sum();
    // calculate wavelet domain threshold based on relative sqrt(total energy)
    let lambda = total_energy.sqrt() as f32 * rel_thresh;

    // perform wavelet thresholding
    swt.decompose(&tmp_buff, &mut t_dom);
    swt.soft_thresh(&mut t_dom,lambda);
    swt.reconstruct(&t_dom,&mut tmp_buff);

    // write filtered image as cfl
    write_cfl(cfl_out,&tmp_buff,vol_d);

}

/// generates the data consistency arrays for low-rank reconstruction. Includes a call to ants registration
/// to correct for eddy-current induced translations
pub fn generate_y(work_dir:impl AsRef<Path>, i:usize, ref_index:usize, vol_dims:&[usize]) {

    let wd = work_dir.as_ref();

    let vol_d = ArrayDim::from_shape(&vol_dims);

    let raw_file = wd.join(format!("raw-{}",i));
    let traj_file = wd.join(format!("traj-{}",i));

    let filtered = wd.join("filtered").join(format!("f-{}",i));
    let out_cfl = wd.join(format!("y-{}",i));
    let out_mask = wd.join(format!("m-{}", i));

    let (raw,raw_dims) = read_cfl(&raw_file);
    let (traj,traj_dims) = read_cfl(&traj_file);
    let (mut g,mask) = grid_cartesian(&raw,raw_dims,&traj,traj_dims,vol_d,true);

    let mask:Vec<_> = mask.into_iter().map(|x|Complex32::new(x,0.)).collect::<Vec<Complex32>>();
    let mask_dims = ArrayDim::from_shape(&vol_dims[1..]);

    let (f,f_dims) = read_cfl(&filtered);
    let f:Vec<_> = f.iter().map(|f|f.norm()).collect::<Vec<_>>();
    write_nifti(&filtered,&f,f_dims);

    if i != ref_index { // this is the reference volume for registration
        let fixed = wd.join("filtered").join(format!("f-{}",ref_index));
        let mov = wd.join("filtered").join(format!("f-{}",i));
        let r = AntsRegistration::translation_only_3d(&fixed,&mov,"_out");
        let trans = r.run_translation().unwrap();
        r.cleanup_outputs().unwrap();
        // modify g to include translation
        g.par_iter_mut().enumerate().for_each(|(i,g)| {
            let [ix,iy,iz,..] = vol_d.calc_idx(i);
            let angle =
                2. * PI * ix as f64 * trans.x / vol_dims[0] as f64 +
                2. * PI * iy as f64* trans.y / vol_dims[1] as f64 +
                2. * PI * iz as f64* trans.z / vol_dims[2] as f64;
            let c = Complex32::from_polar(1., angle as f32);
            *g = *g * c;
            // *g = *g * c.conj(); not sure the correct sign at this point
        });
    }

    fftw_fftn_batched(&mut g,&[vol_dims[0]],vol_dims[1]*vol_dims[2],FftDirection::Inverse,NormalizationType::Unitary);
    g.par_chunks_exact_mut(vol_dims[0]).for_each(|chunk| {
        let a = ArrayDim::from_shape(&[vol_dims[0]]);
        let mut buff = a.alloc(Complex32::ZERO);
        a.fftshift(chunk,&mut buff, true);
        chunk.copy_from_slice(&buff);
    });

    let mut dst = vec![Complex32::ZERO; g.len()];
    let new_dims = permute_left3d(&g, &mut dst, vol_d);
    write_cfl(out_cfl,&dst,new_dims);
    write_cfl(out_mask,&mask,mask_dims);

}

/// generates a phase map for corrections
pub fn generate_phase_maps(work_dir:impl AsRef<Path>, i:usize, vol_dims:&[usize]) {

    let wd = work_dir.as_ref();
    let vol_d = ArrayDim::from_shape(&vol_dims);

    let filtered = wd.join("filtered").join(format!("f-{}",i));
    let phase_out = wd.join(format!("p-{}",i));

    // read filtered, extract phase, permute axes, ifftshift y and z

    let (f,f_dims) = read_cfl(&filtered);
    let phase = f.into_iter().map(|x|{
        Complex32::new(1.,x.to_polar().1)
    }).collect::<Vec<_>>();

    let mut dst = f_dims.alloc(Complex32::ZERO);

    let p_dims = permute_left3d(&phase, &mut dst, vol_d);

    dst.par_chunks_exact_mut(vol_dims[1]*vol_dims[2]).for_each(|yz_slice| {
        let d = ArrayDim::from_shape(&vol_dims[1..]);
        let mut tmp = d.alloc(Complex32::ZERO);
        d.fftshift(yz_slice, &mut tmp, false);
        yz_slice.copy_from_slice(&tmp);
    });

    write_cfl(phase_out,&dst,p_dims);

}


/// permutes the data array by shifting 3 axes to the left by one, moving the x-axis to the third
/// position, the y-axis to the first, and z to the second
pub fn permute_left3d(src:&[Complex32], dst:&mut [Complex32], dims:ArrayDim) -> ArrayDim {

    let shape_orig = dims.shape();
    let x = shape_orig[0];
    let y = shape_orig[1];
    let z = shape_orig[2];

    let mut new_shape = dims.shape().to_vec();
    new_shape[0] = y;
    new_shape[1] = z;
    new_shape[2] = x;

    assert_eq!(x*y*z, dims.numel(),"array must be fully described by 3 dimensions");

    let new_dims = ArrayDim::from_shape(&new_shape);

    dst.par_iter_mut().enumerate().for_each(|(i,s)| {
        let [y,z,x,..] = new_dims.calc_idx(i);
        let addr = dims.calc_addr(&[x,y,z]);
        *s = src[addr];
    });

    new_dims

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