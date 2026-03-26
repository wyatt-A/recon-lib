use array_lib::ArrayDim;
use array_lib::cfl::ndarray::Array3;



pub trait Filter {
    fn filter_coeffs(&self,filter_size:ArrayDim) -> Vec<f32>;
}



/// create a new fermi filter with a cutoff and shape parameter. The cutoff is normalized to
/// the nyquist range `[-0.5,0.5]`
pub struct Fermi {
    cutoff: f32,
    alpha: f32,
}

impl Filter for Fermi {
    fn filter_coeffs(&self,filter_size:ArrayDim) -> Vec<f32> {
        let mut h = filter_size.alloc(0f32);
        h.iter_mut().enumerate().for_each(|(i,u)| {
            // 0 maps to DC
            let [x,y,z,..] = filter_size.calc_idx(i);
            let mut shifted = ArrayDim::dim_buffer_signed();
            let mut norm = ArrayDim::dim_buffer_t::<f32>();
            filter_size.signed_coords(&[x,y,z],&mut shifted);
            norm.iter_mut().zip(shifted.iter()).zip(filter_size.shape()).for_each(|((n,s),d)|{
                *n = *s as f32 / *d as f32;
            });
            let mut r = norm.iter().map(|x| x.powi(2)).sum::<f32>();
            r = r.sqrt();
            *u = 1.0 / (1.0 + ((r - self.cutoff) / self.alpha).exp());
        });
        h
    }
}



impl Fermi {
    pub fn new(cutoff: f32, alpha: f32) -> Fermi {
        Fermi { cutoff, alpha }
    }

}




//let r = (kx*kx + ky*ky).sqrt();
// let h = 1.0 / (1.0 + ((r - kc)/alpha).exp());