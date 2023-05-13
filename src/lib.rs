use opencv::prelude::*;

pub fn convert_to_f32(mat: &Mat) -> opencv::Result<Mat> {
	let mut out = Mat::default();
	mat.convert_to(&mut out, opencv::core::CV_32F, 1.0, 0.0)?;
	Ok(out)
}