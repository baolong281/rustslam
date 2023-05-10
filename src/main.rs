mod extractor;
use extractor::Extractor;
use opencv::core::*;
use opencv::imgproc;
use opencv::imgproc::LINE_8;
use std::env;
use std::process;
use std::path::Path;
use opencv::prelude::*;
use opencv::features2d;
use opencv::{videoio::{
	VideoCapture, 
	CAP_ANY}, };

fn main() {
	//read path 
	let path = env::args()
		.nth(1).unwrap_or_else(|| {
			println!("Error: Missing required argument file path");
			process::exit(1);
		});

	//check if file exists
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

	let matches: Vec<(Point, Point)> = matches.into_iter()
		.map(|(u, v)| {
			let u = u.pt();
			let v = v.pt();
			print!("{:?}   ", u);
			println!("{:?}", v);
			(Point::new(u.x as i32, u.y as i32), Point::new(v.x as i32, v.y as i32))
		})
		.collect();

	matches.into_iter()
		.for_each(|(u, v)| {
			imgproc::line(
				frame,
				u,
				v,
				Scalar::new(0.0, 255.0, 0.0, 0.0),
				1,
				imgproc::LINE_8,
				0
			).unwrap()
		});

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
