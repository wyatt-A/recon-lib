pub mod filters;
pub mod bart_pics;

use std::path::Path;
use array_lib::ArrayDim;
use array_lib::cfl::ndarray::parallel::prelude::*;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use crate::bart_pics::{bart_pics, BartPicsSettings};
use crate::filters::{Fermi, Filter};

#[derive(Debug,Serialize,Deserialize,Clone)]
pub enum ReconMethod {
    CSCartesian{settings:CSCartesianSettings},
    FFT,
}

pub fn run_cs_cartesian(settings:&CSCartesianSettings,work_dir:impl AsRef<Path>, raw_cfl:impl AsRef<Path>, traj_file:impl AsRef<Path>, outfile: impl AsRef<Path>){

    let (raw,raw_dims) = read_cfl(&raw_cfl);
    let (traj,traj_dims) = read_cfl(&traj_file);

    let grid_dims = ArrayDim::from_shape(&settings.grid_dims);

    let g = {
        let filter = settings.filter_coefficients.as_ref().map(|&[a,b]|Fermi::new(a,b));
        let g = grid_cartesian(&raw,raw_dims,&traj,traj_dims,grid_dims, true, filter);
        let mut shifted = grid_dims.alloc(Complex32::ZERO);
        grid_dims.fftshift(&g,&mut shifted,true);
        shifted
    };

    let s = BartPicsSettings::default();
    let (img,..) = bart_pics(&g,grid_dims,&s,&work_dir);

    write_cfl(outfile,&img,grid_dims);

}



impl Default for ReconMethod {
    fn default() -> Self {
        Self::CSCartesian {settings: CSCartesianSettings::default()}
    }
}

#[derive(Debug,Serialize,Deserialize,Clone)]
struct CSCartesianSettings {
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

/// constructs gridded k-space data from compressed views and a trajector mapping for each readout
/// if traj has only 1 entry for dim[0], it is assumed to be 2-D. If it has 2, it is assumed to be 3-D
pub fn grid_cartesian<T:Filter>(data:&[Complex32], data_dims:ArrayDim, traj:&[Complex32], traj_dims:ArrayDim, grid_size:ArrayDim, phase_correct:bool, filter:Option<T> ) -> Vec<Complex32> {

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

    data.chunks_exact(n_read).enumerate().for_each(|(i,view)| {
        let [y,z] = coords[i];
        for x in 0..grid_dims[0] {
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

    vol
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