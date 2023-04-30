use std::env;
use opencv::prelude::*;
use opencv::{videoio::{
	VideoCapture, 
	CAP_ANY}, prelude::Mat};

fn main() {
	let path = env::args()
		.nth(1).unwrap();

	open_video(path).unwrap();

}

fn open_video(path: String) -> opencv::Result<()> {
	let mut capture = VideoCapture::from_file(&path, CAP_ANY )?;

	let mut frame = Mat::default();
    loop {
        capture.read(&mut frame)?;
        if frame.empty() {
            break;
        }
        opencv::highgui::imshow("video", &mut frame)?;
        opencv::highgui::wait_key(25)?;
    }

	Ok(())
}
