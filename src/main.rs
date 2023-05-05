mod extractor;
use extractor::Extractor;
use opencv::core::*;
use std::env;
use std::process;
use std::path::Path;
use opencv::prelude::*;
use opencv::features2d;
use opencv::{videoio::{
	VideoCapture, 
	CAP_ANY}, };

fn main() {
	let path = env::args()
		.nth(1).unwrap_or_else(|| {
			println!("Error: Missing required argument file path");
			process::exit(1);
		});

	check_file(&path).unwrap_or_else(|_| {
			println!("Error: File invalid or does not exist");
			process::exit(1);
	});

	let mut capture = VideoCapture::from_file(&path, CAP_ANY ).unwrap();

	let mut frame = Mat::default();
	let mut ext = Extractor::new().unwrap_or_else(|e| {
		println!("{}", e);
		process::exit(1);
	});

	loop {
		capture.read(&mut frame).unwrap();
			if frame.empty() {
				break;
		}

		process_frame(&mut frame, &mut ext)
			.unwrap_or_else(|e| {
				println!("{}", e);
				process::exit(1);
			});

		}
	process::exit(1); 
}

fn process_frame(frame: &mut Mat, ext: &mut Extractor) -> opencv::Result<()> {
	let mut img = Mat::default();
	let (last_img, last_kps, last_desc) = ext.get_last();

	ext.extract(frame.clone())
		.unwrap_or_else(|e| {
			println!("{}", e);
			process::exit(1);
		});

	//features2d::draw_matches(last_img, last_kps, );

	opencv::highgui::imshow("rustslam", &img)?;
	opencv::highgui::wait_key(400)?;
	Ok(())
}


fn check_file(path: &String) -> Result<(), ()>{
	if !Path::new(&path).exists() {
		return Err(());
	}
	Ok(())
}
