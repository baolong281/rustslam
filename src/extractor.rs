use opencv::core::*;
use opencv::features2d;
use opencv::features2d:: {
	ORB, ORB_ScoreType, BFMatcher
};
use opencv::prelude::DescriptorMatcherConst;

struct LastData {
	img: Mat,
	kps: Vector<KeyPoint>,
	desc: Mat,
}

pub struct Extractor {
	orb: Box<dyn ORB>,
	matcher: BFMatcher,
	last: LastData,
}

impl Extractor {
	pub fn new() -> opencv::Result<Extractor>{

		let orb = Box::new(<dyn ORB>::create(
			2000,
			1.,
			8,
			15,
			0,
			2,
			ORB_ScoreType::HARRIS_SCORE,
			31,
			20,
		)?);

		let matcher = BFMatcher::new(NORM_L2, false)?;
		let img = Mat::default();
		let kps = Vector::default();
		let desc = Mat::default();

		Ok(Extractor { orb, matcher , last: LastData {img , kps, desc}})
	}

	pub fn extract(&mut self, frame: Mat) -> opencv::Result<()> {
		let mut orb_keypoints = Vector::default();
		let mut orb_desc = Mat::default();
		let mut dst_img = Mat::default();
		let mask = Mat::default();
		self.orb.detect_and_compute(&frame, &mask, &mut orb_keypoints, &mut orb_desc, false)?;
		features2d::draw_keypoints(
			&frame,
			&orb_keypoints,
			&mut dst_img,
			VecN([0., 255., 0., 255.]),
			features2d::DrawMatchesFlags::DEFAULT,
		)?;

		let mut knn: Vector<DMatch> = Vector::default();

		self.matcher.train_match(&orb_desc, &self.last.desc, &mut knn, &mask)?;
		self.last.img = frame;
		self.last.kps = orb_keypoints;
		self.last.desc = orb_desc;

		Ok(())
	}

	pub fn get_last(&self) -> (&Mat, &Vector<KeyPoint>, &Mat) {
		(&self.last.img, &self.last.kps, &self.last.desc)
	}
}


