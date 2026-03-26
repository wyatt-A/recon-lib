use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::Complex32;
use recon_lib::bart_pics::{bart_pics, BartPicsSettings};
use recon_lib::filters::Fermi;
use recon_lib::grid_cartesian;

fn main() {


    let (raw,raw_dims) = read_cfl("/Users/Wyatt/object-manager/raw_148.cfl");
    let (traj,traj_dims) = read_cfl("/Users/Wyatt/object-manager/traj_148.cfl");

    let grid_dims = ArrayDim::from_shape(&[512,256,256]);

    let f = Fermi::new(0.48,0.03);

    let g = {
        let g = grid_cartesian::<Fermi>(&raw,raw_dims,&traj,traj_dims,grid_dims, true, Some(f));
        let mut shifted = grid_dims.alloc(Complex32::ZERO);
        grid_dims.fftshift(&g,&mut shifted,true);
        shifted
    };

    let s = BartPicsSettings::default();
    let (img,..) = bart_pics(&g,grid_dims,&s,".");

    let img = img.iter().map(|x| x.norm()).collect::<Vec<f32>>();
    write_nifti("img1.nii",&img,grid_dims);
    //write_cfl("img",&g,grid_dims);

}