use std::path::PathBuf;
use array_lib::ArrayDim;
use array_lib::io_cfl::read_cfl;
use array_lib::io_nifti::write_nifti;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    work_dir:PathBuf,
    n:usize,
    file_prefix: String,
    output: String,
}

fn main() {

    let args = Args::parse();


    let to_load = args.work_dir.join(format!("{}-{}",args.file_prefix,0));
    let (..,dims) = read_cfl(&to_load);
    let shape = dims.shape();

    let offset = shape[2] / 2;

    let buff_dims = ArrayDim::from_shape(&[shape[0], shape[1], args.n]);
    let mut buff = buff_dims.alloc(0f32);

    buff.chunks_exact_mut(shape[0] * shape[1]).enumerate().for_each(|(i,slice)| {
        println!("loading {}",i);
        let to_load = args.work_dir.join(format!("{}-{}",args.file_prefix,i));
        let (data,..) = read_cfl(to_load);
        let start = offset * shape[0] * shape[1];
        let end = start + shape[0] * shape[1];
        let s = &data[start..end];
        let s:Vec<_> = s.iter().map(|x| x.norm()).collect();
        slice.copy_from_slice(&s);
    });

    write_nifti(args.work_dir.join(args.output).with_extension("nii"),&buff,buff_dims);


}