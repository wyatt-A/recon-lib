use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use ants_reg_wrapper::AntsRegistration;
use array_lib::io_cfl::read_cfl;
use array_lib::io_nifti::write_nifti;
use clap::Parser;

#[derive(Parser)]
struct Args {
    work_dir:PathBuf,
    n:usize,
}


fn main() {
    let args = Args::parse();
    run_reg(args.work_dir,args.n);
}

/// finds linear translations to register images for low-rank recon
fn run_reg(work_dir:impl AsRef<Path>, n:usize) {

    let filtered = work_dir.as_ref().join(format!("f-{}",0));
    let ref_nii = filtered.with_extension("nii");
    let (data,dims) = read_cfl(&filtered);
    let x:Vec<_> = data.into_iter().map(|x|x.norm()).collect();
    write_nifti(&ref_nii,&x,dims);

    let mut trans_vox = vec![];
    trans_vox.push([0.,0.,0.]);

    for i in 1..n {
        let filtered = work_dir.as_ref().join(format!("f-{}",i));
        let nii = filtered.with_extension("nii");
        let (data,dims) = read_cfl(&filtered);
        let x:Vec<_> = data.into_iter().map(|x|x.norm()).collect();
        write_nifti(&nii,&x,dims);
        let r = AntsRegistration::translation_only_3d(&ref_nii,&nii,"out_");
        let trans = r.run_translation().unwrap();
        r.cleanup_outputs().unwrap();
        trans_vox.push([trans.x,trans.y,trans.z]);
        fs::remove_file(nii).unwrap();
    }

    let lines:Vec<_> = trans_vox.iter().map(|t| format!("{} {} {}",t[0],t[1],t[2])).collect();
    let s = lines.join("\n");
    let mut f = File::create(work_dir.as_ref().join("trans.txt")).unwrap();
    f.write_all(s.as_bytes()).unwrap();

}