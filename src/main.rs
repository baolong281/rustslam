mod extractor;
use extractor::Extractor;
use opencv::core::Point;
use opencv::core::Point_;
use opencv::core::Scalar;
use opencv::imgproc::circle;
use std::env;
use std::process;
use std::path::Path;
use opencv::prelude::*;
use opencv::{videoio::{
	VideoCapture, 
	CAP_ANY}, prelude::Mat};

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
	let ext = Extractor::new();

	loop {
		capture.read(&mut frame).unwrap();
			if frame.empty() {
				break;
		}

		let mut feats = ext.extract(frame.clone())
			.unwrap_or_else(|e| {
				println!("{}", e);
				process::exit(1);
			});

		process_frame(&mut frame, &mut feats)
			.unwrap_or_else(|e| {
				println!("{}", e);
				process::exit(1);
			});
		}
}

fn process_frame(frame: &mut Mat, feats: &mut Mat) -> opencv::Result<()> {

	let feature_size = feats.rows();
	println!("FEATS SIZE: {feature_size}");

	for i in 0..feature_size {
		let b =  feats.at_nd::<Point_<f32>>(&[i, 0])?;
		let point: Point_<i32> = Point::new(b.x as i32, b.y as i32);
		draw_circle(frame, point)?;
	}

	opencv::highgui::imshow("video", frame)?;
	opencv::highgui::wait_key(400)?;
	Ok(())
}

fn draw_circle(frame: &mut Mat, center: Point_<i32>) -> opencv::Result<()> {
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let radius = 3;
    circle(frame, center, radius, color, 1, 8, 0)?;
	Ok(())
}

fn check_file(path: &String) -> Result<(), ()>{
	if !Path::new(&path).exists() {
		return Err(());
	}
	Ok(())
}
