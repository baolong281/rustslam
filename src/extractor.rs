use opencv::{core::*, imgproc, calib3d};
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
    last: Option<LastData>,
}

impl Extractor {
    pub fn new() -> opencv::Result<Extractor>{

        let orb = Box::new(<dyn ORB>::create(
            3000,
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

        Ok(Extractor { orb, matcher , last: None })
    }

    pub fn extract(&mut self, frame: &Mat) -> opencv::Result<Vec<(Point, Point)>> {

        //convert to grayscale and get features
        let mut gray = Mat::default();
        let mut feats = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 1)?;
        imgproc::good_features_to_track(
            &gray,
            &mut feats,
            3000,
            0.01,
            3.0,
            &no_array(),
            3,
            false,
            0.04)?;

        //turn features mat into vector of keypoints
        let mut kps = feats
            .iter::<Point2f>()?
            .map(|(_, point)| {
                KeyPoint::new_point(point, 20., -1., 0., 0, -1).unwrap()
            })
            .collect::<Vector<KeyPoint>>();


        //computing descriptors 
        let mut orb_desc = Mat::default();
        let mask = Mat::default();
        self.orb.compute(&gray, &mut kps, &mut orb_desc)?;

        //run the matche if last is some
        let mut matches_vec= Vector::default();
        let mut matches: Vec<(Point, Point)> = vec![];
        if let Some(last) = &mut self.last {
            self.matcher.knn_train_match(&orb_desc, &last.desc, &mut matches_vec, 2, &mask, false)?;

            //convert vector of matches to tuples of both keypoints
            matches = matches_vec 
                .iter()
                .filter(|m1| {
                    //filter bad matches using distance
                    m1.get(0).unwrap().distance < m1.get(1).unwrap().distance * 0.75
                })
                .filter(|m1| {
                    //remove matches with out of bounds idk why
                    m1.get(0).unwrap().train_idx < kps.len() as i32 && m1.get(0).unwrap().query_idx < kps.len() as i32
                })
                .map(|x| {
                    //maps into the keypoint
                    (kps.get(x.get(0).unwrap().query_idx as usize).unwrap(),
                    last.kps.get(x.get(0).unwrap().train_idx as usize).unwrap()
                )
                })
                .map(|(u, v)| {
                    let u = u.pt();
                    let v = v.pt();
                    (Point::new(u.x as i32, u.y as i32), Point::new(v.x as i32, v.y as i32))
                })
                .collect();

            //usac and matrix transform
            let p1: Vector<Point> = Vector::from_iter(matches.iter().map(|(u, _)| u).copied());
            let p2: Vector<Point> = Vector::from_iter(matches.iter().map(|(_, v)| v).copied());
            let mut inliners = Mat::default();
            let _ = calib3d::find_fundamental_mat(&p1, &p2, calib3d::USAC_FM_8PTS, 1.0, 0.99, 100, &mut inliners)?;
            let inliners = inliners.to_vec_2d::<u8>()?
                .concat()
                .iter().
                map(|v| *v!= 0)
                .collect::<Vec<_>>();

            let p1: Vec<Point> = p1.iter().zip(&inliners).filter(|(_, x) | **x).map(|(u, _)| u).collect();
            let p2: Vec<Point> = p2.iter().zip(&inliners).filter(|(_, x) | **x).map(|(u, _)| u).collect();
            
            matches = p1.into_iter().zip(p2.into_iter()).collect();

            last.img = frame.clone();
            last.kps = kps.clone();
            last.desc = orb_desc.clone();

        } else {
            //declare last if last is none
            self.last = Some(LastData {img: frame.clone(), kps: kps.clone(), desc: orb_desc.clone()});
        }

        Ok((matches))
    }

    pub fn get_last(&self) -> Option<(Mat, Vector<KeyPoint>, Mat)> {
        match &self.last {
            Some(last) => {
                Some((last.img.clone(), last.kps.clone(), last.desc.clone()))
            }
            None => None
        }
    }

    pub fn check_last(&self) -> Option<()> {
        match &self.last {
            Some(last) => {
                Some(())
            }
            None => None
        }
    }
}


