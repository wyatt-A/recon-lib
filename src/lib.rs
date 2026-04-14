pub mod filters;
pub mod bart_pics;

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use array_lib::ArrayDim;
use array_lib::cfl::ndarray::parallel::prelude::*;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::rs_fft::rs_fftn;
use dwt_lib::swt3::SWT3Plan;
use dwt_lib::wavelet::{Wavelet, WaveletType};
use serde::{Deserialize, Serialize};
use serde::__private228::de::missing_field;
use crate::bart_pics::{bart_pics, BartPicsSettings};
use crate::filters::{Fermi, Filter};

#[derive(Debug,Serialize,Deserialize,Clone)]
pub enum ReconMethod {
    CSCartesian{settings:CSCartesianSettings},
    FFT,
}

pub fn run_cs_cartesian(settings:&CSCartesianSettings, work_dir:impl AsRef<Path>, raw_cfl:impl AsRef<Path>, traj_file:impl AsRef<Path>, outfile: impl AsRef<Path>) {
    let (raw,raw_dims) = read_cfl(&raw_cfl);
    let (traj,traj_dims) = read_cfl(&traj_file);
    let grid_dims = ArrayDim::from_shape(&settings.grid_dims);
    let g = {
        let filter = settings.filter_coefficients.as_ref().map(|&[a,b]|Fermi::new(a,b));
        let (g,_) = grid_cartesian_f(&raw, raw_dims, &traj, traj_dims, grid_dims, true, filter);
        let mut shifted = grid_dims.alloc(Complex32::ZERO);
        grid_dims.fftshift(&g,&mut shifted,true);
        shifted
    };
    let s = &settings.bart_settings;
    let (img,..) = bart_pics(&g,grid_dims,s,&work_dir);
    write_cfl(outfile,&img,grid_dims);
}



impl Default for ReconMethod {
    fn default() -> Self {
        Self::CSCartesian {settings: CSCartesianSettings::default()}
    }
}

#[derive(Debug,Serialize,Deserialize,Clone)]
pub struct CSCartesianSettings {
    bart_settings: BartPicsSettings,
    grid_dims: [usize;3],
    filter_coefficients: Option<[f32;2]>,
}

impl Default for CSCartesianSettings {
    fn default() -> Self {
        Self {
            grid_dims: [512,256,256],
            bart_settings: BartPicsSettings::default(),
            filter_coefficients: None,
        }
    }
}

impl CSCartesianSettings {
    pub fn to_file(&self,filename:impl AsRef<Path>) {
        let s = serde_json::to_string_pretty(&self).unwrap();
        let mut f = File::create(filename.as_ref().with_extension("json")).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    pub fn from_file(filename:impl AsRef<Path>) -> Self {
        let mut f = File::open(filename.as_ref().with_extension("json")).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        serde_json::from_str(&s).unwrap()
    }

    pub fn to_toml_table(&self) -> toml::Table {
        let value = toml::Value::try_from(self).unwrap();
        value.as_table().unwrap().to_owned()
    }

}



pub fn estimate_phase_mask(img:&[Complex32],phase:&mut [Complex32], dims:ArrayDim, rel_thresh:f32, n_levels:usize) {

    let shape = dims.shape_squeeze();
    let swt3 = SWT3Plan::new(&shape[0..3],n_levels,Wavelet::new(WaveletType::Daubechies2));
    let mut t = vec![Complex32::ZERO; swt3.t_domain_size()];
    swt3.decompose(img, &mut t);
    swt3.soft_thresh(&mut t, rel_thresh);
    swt3.reconstruct(&t,phase);

    phase.iter_mut().for_each(|x|{
        *x /= x.norm();
    })


}


pub fn grid_cartesian(data:&[Complex32], data_dims:ArrayDim, traj:&[Complex32], traj_dims:ArrayDim, grid_size:ArrayDim, phase_correct:bool ) -> (Vec<Complex32>,Vec<f32>) {

    let data_dims = data_dims.shape();
    let grid_dims = grid_size.shape();
    let coords = traj_to_coords(traj,traj_dims);
    let n_read = data_dims[0];
    let n_views = data_dims[1];
    assert_eq!(n_views, coords.len());
    let mut vol = grid_size.alloc(Complex32::ZERO);

    let mut max_energy = 0.0;
    let mut max_energy_coords = [0,0,0];

    let mask_dims = ArrayDim::from_shape(&grid_dims[1..]);

    let mut mask = mask_dims.alloc(0f32);

    data.chunks_exact(n_read).enumerate().for_each(|(i,view)| {
        let [y,z] = coords[i];
        for x in 0..grid_dims[0] {

            let mask_addr = mask_dims.calc_addr_signed(&[y,z]);
            mask[mask_addr] = 1.;

            let addr = grid_size.calc_addr_signed(&[x as isize, y, z]);
            let s = view.get(x).cloned().unwrap_or(Complex32::ZERO);
            if phase_correct {
                let e = s.norm_sqr();
                if e > max_energy {
                    max_energy = e;
                    max_energy_coords = [x as isize,y,z];
                }
            }
            vol[addr] = s;
        }
    });

    let (vol,mask) = if phase_correct {

        let phase_addr = grid_size.calc_addr_signed(&max_energy_coords);
        let phase = vol[phase_addr].to_polar().1;
        vol.par_iter_mut().for_each(|x| *x = *x * Complex32::from_polar(1.,-phase));

        let mut shifted = grid_size.alloc(Complex32::ZERO);
        let mut mask_shifted = mask_dims.alloc(0f32);
        let [dx,dy,dz] = max_energy_coords;
        let shift = [-dx,-dy,-dz];
        let mask_shift = [-dy,-dz];
        grid_size.circshift(&shift,&vol,&mut shifted);
        mask_dims.circshift(&mask_shift,&mask,&mut mask_shifted);
        (shifted,mask_shifted)
    }else {
        (vol,mask)
    };

    (vol,mask)
}


/// constructs gridded k-space data from compressed views and a trajector mapping for each readout
/// if traj has only 1 entry for dim[0], it is assumed to be 2-D. If it has 2, it is assumed to be 3-D
pub fn grid_cartesian_f<T:Filter>(data:&[Complex32], data_dims:ArrayDim, traj:&[Complex32], traj_dims:ArrayDim, grid_size:ArrayDim, phase_correct:bool, filter:Option<T> ) -> (Vec<Complex32>, Vec<f32>) {

    // if a filter is passed, phase correction is set to true
    let phase_correct = if filter.is_some() {
        true
    }else {
        phase_correct
    };

    let data_dims = data_dims.shape();
    let grid_dims = grid_size.shape();
    let coords = traj_to_coords(traj,traj_dims);
    let n_read = data_dims[0];
    let n_views = data_dims[1];
    assert_eq!(n_views, coords.len());
    let mut vol = grid_size.alloc(Complex32::ZERO);

    let mut max_energy = 0.0;
    let mut max_energy_coords = [0,0,0];

    let mask_dims = ArrayDim::from_shape(&grid_dims[1..]);

    let mut mask = mask_dims.alloc(0f32);

    data.chunks_exact(n_read).enumerate().for_each(|(i,view)| {
        let [y,z] = coords[i];
        for x in 0..grid_dims[0] {

            let mask_addr = mask_dims.calc_addr_signed(&[y,z]);
            mask[mask_addr] = 1.;

            let addr = grid_size.calc_addr_signed(&[x as isize, y, z]);
            let s = view.get(x).cloned().unwrap_or(Complex32::ZERO);
            if phase_correct {
                let e = s.norm_sqr();
                if e > max_energy {
                    max_energy = e;
                    max_energy_coords = [x as isize,y,z];
                }
            }
            vol[addr] = s;
        }
    });

    let mut vol = if phase_correct {
        let mut shifted = grid_size.alloc(Complex32::ZERO);
        let [dx,dy,dz] = max_energy_coords;
        let shift = [-dx,-dy,-dz];
        grid_size.circshift(&shift,&vol,&mut shifted);
        shifted
    }else {
        vol
    };

    if let Some(filter) = filter {
        let h = filter.filter_coeffs(grid_size);
        vol.par_iter_mut().zip(h.par_iter()).for_each(|(x,h)|{
            *x = x.scale(*h);
        })
    }

    (vol,mask)
}

fn traj_to_coords(traj:&[Complex32], traj_dims:ArrayDim) -> Vec<[isize;2]> {
    assert_eq!(traj_dims.numel(), traj.len());
    let traj_shape = traj_dims.shape();
    (0..traj_shape[1]).map(|i|{
        let y = traj[traj_dims.calc_addr(&[0,i])].re as isize;
        let z = if traj_shape[0] > 1 {
            traj[traj_dims.calc_addr(&[1,i])].re as isize
        }else {
            0
        };
        [y,z]
    }).collect()
}

/// finds a normalization factor based on the k-space calibration region. Dividing by the returned
/// scale results in a normalized image. k-space is assumed to be already phase corrected. saturation
/// fraction should be a small value representing the most intense values
pub fn signal_scale(ksp:&[Complex32], dims:ArrayDim, calib_size:&[usize], sat_frac:f32) -> f32 {
    assert!((0.0..1.0).contains(&sat_frac),"saturation fraction must be between 0 and 1");
    let calib_dims = ArrayDim::from_shape(calib_size);
    assert!(calib_dims.numel() <= dims.numel(), "calibration size should be less than or equal to ksp");
    let mut y = calib_dims.alloc(Complex32::ZERO);
    // copy calibration region from ksp
    y.iter_mut().enumerate().for_each(|(i,y)| {
        let [ix,iy,iz,..] = calib_dims.calc_idx(i);
        let mut signed = [0,0,0];
        calib_dims.signed_coords(&[ix,iy,iz],&mut signed);
        let addr = dims.calc_addr_signed(&signed);
        *y = ksp[addr];
    });
    // go to image domain
    rs_fftn(&mut y,calib_size,FftDirection::Inverse,NormalizationType::Unitary);
    // reverse sort, largest to smallest by norm
    y.sort_by(|a,b| b.norm().partial_cmp(&a.norm()).unwrap());
    // retrieve the first sample below the saturation band. This guards against a few bright samples
    let idx = ((y.len() as f32 * sat_frac).floor() as usize).clamp(0,y.len()-1);
    y[idx].norm()
}