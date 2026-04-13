use std::f64::consts::PI;
use std::fs::{create_dir_all, read};
use std::path::{Path, PathBuf};
use ants_reg_wrapper::AntsRegistration;
use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, read_cfl_slice, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::fftw_fft::{fftw_fftn, fftw_fftn_batched};
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use lr_rs::rs_svd::{svd_hard, svd_soft};
use rayon::prelude::*;
use recon_lib::grid_cartesian;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    sub_cmd: SubCmd,
}

#[derive(Args, Debug, Clone)]
pub struct FilterArgs {
    work_dir:PathBuf,
    index:usize,
    x:usize,
    y:usize,
    z:usize,
}

#[derive(Args, Debug, Clone)]
pub struct GenYArgs {
    work_dir:PathBuf,
    index:usize,
    ref_index:usize,
    x:usize,
    y:usize,
    z:usize,
}

#[derive(Args, Debug, Clone)]
pub struct PhaseArgs {
    work_dir:PathBuf,
    index:usize,
    x:usize,
    y:usize,
    z:usize,
}

#[derive(Args, Debug, Clone)]
pub struct PrepIterateArgs {
    work_dir:PathBuf,
    index:usize,
    radius:usize,
    n_vols:usize,
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
    PrepIterate(PrepIterateArgs),
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
        SubCmd::PrepIterate(args) => {
            generate_iterates(args.work_dir, args.index, args.radius, args.n_vols, &[args.x,args.y,args.z])
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

    // don't apply shift to reference volume (usually volume 0)
    if i != ref_index {
        let fixed = wd.join("filtered").join(format!("f-{}",ref_index)).with_extension("nii");
        let mov = wd.join("filtered").join(format!("f-{}",i)).with_extension("nii");;
        let r = AntsRegistration::translation_only_3d(&fixed,&mov,"_out");
        let trans = r.run_translation().unwrap();
        r.cleanup_outputs().unwrap();
        println!("applying translation: x:{}, y:{}, z:{}",trans.x,trans.y,trans.z);
        // modify g to include translation
        g.par_iter_mut().enumerate().for_each(|(i,g)| {
            let mut signed = [0,0,0];
            let [ix,iy,iz,..] = vol_d.calc_idx(i);
            vol_d.signed_coords(&[ix,iy,iz],&mut signed);
            let dx = signed[0] as f64;
            let dy = signed[1] as f64;
            let dz = signed[2] as f64;
            let angle =
                2. * PI * dx * trans.x / vol_dims[0] as f64 +
                2. * PI * dy * trans.y / vol_dims[1] as f64 +
                2. * PI * dz * trans.z / vol_dims[2] as f64;
            let c = Complex32::from_polar(1., angle as f32);
            *g = *g * c.conj();
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
    let new_dims = vol_d.permute(&g,&mut dst,&[1,2,0]);
    write_cfl(out_cfl,&dst,new_dims);
    write_cfl(out_mask,&mask,mask_dims);

}

/// generates a phase map for corrections
pub fn generate_phase_maps(work_dir:impl AsRef<Path>, i:usize, vol_dims:&[usize]) {
    let wd = work_dir.as_ref();
    let vol_d = ArrayDim::from_shape(&vol_dims);
    let filtered = wd.join("filtered").join(format!("f-{}",i));
    let phase_out = wd.join(format!("p-{}",i));
    // read filtered, extract phase, permute axes, inverse fft shift y and z
    let (f,f_dims) = read_cfl(&filtered);
    let phase = f.into_iter().map(|x|{
        Complex32::from_polar(1.,x.to_polar().1)
    }).collect::<Vec<_>>();
    let mut dst = f_dims.alloc(Complex32::ZERO);
    let p_dims = vol_d.permute(&phase, &mut dst, &[1,2,0]);
    dst.par_chunks_exact_mut(vol_dims[1]*vol_dims[2]).for_each(|yz_slice| {
        let d = ArrayDim::from_shape(&vol_dims[1..]);
        let mut tmp = d.alloc(Complex32::ZERO);
        d.fftshift(yz_slice, &mut tmp, false);
        yz_slice.copy_from_slice(&tmp);
    });
    write_cfl(phase_out,&dst,p_dims);
}


pub fn generate_iterates(work_dir:impl AsRef<Path>, center_slice:usize, radius:usize, n_vols:usize, vol_dims:&[usize]) {

    let wd = work_dir.as_ref();

    let (it_y,it_dims) = prep_iterate_y(&work_dir, center_slice, radius, n_vols, vol_dims);
    let (it_phase,..) = prep_iterate_phase(&work_dir, center_slice, radius, n_vols, vol_dims);
    let (it_mask,mask_dims) = prep_iterate_mask(&work_dir, n_vols, vol_dims);

    write_cfl(wd.join(format!("it-y-{}",center_slice)),&it_y,it_dims);
    write_cfl(wd.join(format!("it-p-{}",center_slice)),&it_phase,it_dims);
    write_cfl(wd.join(format!("it-m-{}",center_slice)),&it_mask,mask_dims);
}



pub fn prep_iterate_y(work_dir:impl AsRef<Path>, center_slice:usize, radius:usize, n_vols:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {

    let wd = work_dir.as_ref();

    let slice_stride = vol_dims[1] * vol_dims[2];
    let slab_stride = slice_stride * (radius*2 + 1);

    let y_dims = ArrayDim::from_shape(&[vol_dims[1], vol_dims[2], radius*2 + 1, n_vols]);
    let mut y = y_dims.alloc(Complex32::ZERO);

    let xd = ArrayDim::from_shape(&vol_dims[0..1]);
    let ci = center_slice as isize;
    let s_start = ci - radius as isize;
    let s_end = ci + radius as isize;
    let indices = (s_start..=s_end).collect::<Vec<isize>>();

    y.chunks_exact_mut(slab_stride).enumerate().for_each(|(i,y)|{
        y.chunks_exact_mut(slice_stride).enumerate().for_each(|(j,y)|{
            let x_addr = xd.calc_addr_signed(&[indices[j]]);
            let f = wd.join(format!("y-{}",i));
            println!("loading {}",f.display());
            read_cfl_slice(f,x_addr * slice_stride,y);
        });
    });

    (y,y_dims)

}

pub fn prep_iterate_phase(work_dir:impl AsRef<Path>, center_slice:usize, radius:usize, n_vols:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {
    let wd = work_dir.as_ref();

    let slice_stride = vol_dims[1] * vol_dims[2];
    let slab_stride = slice_stride * (radius*2 + 1);

    let p_dims = ArrayDim::from_shape(&[vol_dims[1], vol_dims[2], radius*2 + 1, n_vols]);
    let mut p = p_dims.alloc(Complex32::ZERO);

    let xd = ArrayDim::from_shape(&vol_dims[0..1]);
    let ci = center_slice as isize;
    let s_start = ci - radius as isize;
    let s_end = ci + radius as isize;
    let indices = (s_start..s_end).collect::<Vec<isize>>();

    p.chunks_exact_mut(slab_stride).enumerate().for_each(|(i,p)|{
        p.chunks_exact_mut(slice_stride).enumerate().for_each(|(j,p)|{
            let x_addr = xd.calc_addr_signed(&[indices[j]]);
            read_cfl_slice(wd.join(format!("p-{}",i)),x_addr * slice_stride,p);
        });
    });

    (p, p_dims)
}

pub fn prep_iterate_mask(work_dir:impl AsRef<Path>, n_vols:usize, vol_dims:&[usize]) -> (Vec<Complex32>, ArrayDim) {
    let wd = work_dir.as_ref();
    let slice_stride = vol_dims[1] * vol_dims[2];
    let m_dims = ArrayDim::from_shape(&[vol_dims[1],vol_dims[2],n_vols]);
    let mut m = m_dims.alloc(Complex32::ZERO);
    m.chunks_exact_mut(slice_stride).enumerate().for_each(|(i,m)|{
        read_cfl_slice(wd.join(format!("m-{}",i)),0,m);
    });
    (m,m_dims)
}

