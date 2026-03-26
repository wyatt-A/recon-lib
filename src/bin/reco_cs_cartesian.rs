use std::path::{Path, PathBuf};
use clap::Parser;
use recon_lib::{run_cs_cartesian, CSCartesianSettings};

#[derive(Parser, Debug)]
struct Args {
    reco_settings: PathBuf,
    work_dir: PathBuf,
    raw_cfl: PathBuf,
    traj_cfl: PathBuf,
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let settings = CSCartesianSettings::from_file(args.reco_settings);
    run_cs_cartesian(
        &settings,
        &args.work_dir,
        args.work_dir.join(&args.raw_cfl),
        args.work_dir.join(&args.traj_cfl),
        args.work_dir.join(&args.output)
    )
}