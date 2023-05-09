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
	
	let (img_1, kps_1, _) = match ext.get_last() {
		Some(f) => f,
		None => {
			ext.extract(frame.clone())?;
			return Ok(());
		}
	};

	let (img_2, kps_2, matches) = ext.extract(frame.clone())?;

	let keypoints = matches
		.into_iter()
		.map(|(x, y)| {
			x
		})
		.collect();

	features2d::draw_keypoints(&frame.clone(), &keypoints, frame, Scalar::all(-1.0), features2d::DrawMatchesFlags::DEFAULT)?;

	opencv::highgui::imshow("rustslam", frame)?;
	opencv::highgui::wait_key(400)?;

	//println!("{:?}", matches);
	Ok(())
}


fn check_file(path: &String) -> Result<(), ()>{
	if !Path::new(&path).exists() {
		return Err(());
	}
	Ok(())
}
