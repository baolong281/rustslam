mod extractor;
use extractor::Extractor;
use opencv::core::*;
use opencv::imgproc;
use rustslam::convert_to_f32;
use std::env;
use std::process;
use std::path::Path;
use opencv::prelude::*;
use opencv::{videoio::{
        VideoCapture, 
        CAP_ANY}, };

#[allow(non_snake_case)]
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
    let mut out = Mat::default();
    let res = Size_::new(1920, 1080);

    //intrinsic matrix
    let K = get_k(&res).unwrap_or_else(|e| {
        println!("{}", e);
        process::exit(1);
    });

    let mut ext = Extractor::new(K).unwrap_or_else(|e| {
        println!("{}", e);
        process::exit(1);
    });

    loop {
        capture.read(&mut frame).unwrap();
            if frame.empty() {
                break;
        }

        imgproc::resize(&frame, &mut out, res, 0.0, 0.0, imgproc::INTER_LINEAR).unwrap_or_else(|_| println!("FORTNITE"));

        process_frame(&mut out, &mut ext)
            .unwrap_or_else(|e| {
                println!("{}", e);
                process::exit(1);
            });

    }
    process::exit(1); 
}

fn process_frame(frame: &mut Mat, ext: &mut Extractor) -> opencv::Result<()> {

    match ext.check_last() {
        Some(f) => f,
        None => {
            ext.extract(&frame)?;
            return Ok(());
        }
    };
    let matches = ext.extract(&frame)?;

    matches.into_iter()
        .for_each(|(mut u, mut v)| {
			ext.denormalize_point(&mut u);
			ext.denormalize_point(&mut v);
            imgproc::line(
                frame,
                u,
                v,
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0
            ).unwrap();
            imgproc::circle(frame, v, 3, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0).unwrap();
            imgproc::circle(frame, u, 3, Scalar::new(0.0, 0.0, 255.0, 0.0), 1, imgproc::LINE_8, 0).unwrap();
			
        });

    opencv::highgui::imshow("rustslam", frame)?;
    opencv::highgui::wait_key(1)?;

    //println!("{:?}", matches);
    Ok(())
}


fn check_file(path: &String) -> Result<(), ()>{
    if !Path::new(&path).exists() {
        return Err(());
    }
    Ok(())
}

//get intrinsic matrix
#[allow(non_snake_case)]
fn get_k(res: &Size_<i32>) -> opencv::Result<Mat> {
    //focal length
    let F = 1;
    //intrinsic matrix
    let K = Mat::from_slice_2d(&vec![
        vec![F, 0, res.width/2],
        vec![0, F, res.height/2],
        vec![0, 0, 1],
        ]).unwrap_or_else(|e| {
            println!("{:?}", e);
            process::exit(1);
        });
	let K = convert_to_f32(&K)?;

    Ok(K)
}
