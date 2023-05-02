use opencv::imgproc;
use opencv::core::*;

pub struct Extractor;

impl Extractor {

	pub fn new() -> Extractor {
		Extractor {}
	}

	pub fn extract(&self, frame: Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    let mut feats = Mat::default();
    imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 1)?;
    imgproc::good_features_to_track(&gray, &mut feats, 3000, 0.01, 3.0, &no_array(), 3, false, 0.04)?;
    Ok(feats)
	}
}
