use opencv::{core::*, imgproc, calib3d, features2d};
use opencv::features2d:: {
    ORB, ORB_ScoreType, BFMatcher
};
use opencv::prelude::DescriptorMatcherConst;

struct LastData {
    kps: Vector<KeyPoint>,
    desc: Mat,
}

pub struct Extractor {
    orb: Box<dyn ORB>,
    matcher: BFMatcher,
    last: Option<LastData>,
    gray: Mat,
    kps: Vector<KeyPoint>,
    mask: Mat,
    orb_desc: Mat,
}

impl Extractor {
    pub fn new() -> opencv::Result<Extractor>{
        let orb = Box::new(<dyn ORB>::create(
            3000,
            1.2, 
            8,
            16,
            0,
            2,
            ORB_ScoreType::FAST_SCORE,
            31,
            20,
        )?);

        let matcher = BFMatcher::new(NORM_L2, false)?;
        let gray = Mat::default();
        let kps= Vector::default();
        let mask = Mat::default();
        let orb_desc = Mat::default();

        Ok(Extractor { orb, matcher , last: None, gray, kps, mask, orb_desc })
   }

    pub fn extract(&mut self, frame: &Mat) -> opencv::Result<Vec<(Point, Point)>> {
        //convert to grayscale and get features
        imgproc::cvt_color(frame, &mut self.gray, imgproc::COLOR_BGR2GRAY, 1)?;
        features2d::fast(&self.gray, &mut self.kps, 20, true)?;

        //computing descriptors 
        self.orb.compute(&self.gray, &mut self.kps, &mut self.orb_desc)?;

        //run the matche if last is some
        let mut matches: Vec<(Point, Point)> = vec![];
        if let Some(last) = &mut self.last {
            //get matches
            let mut matches_vec= Vector::default();
            self.matcher.knn_train_match(&self.orb_desc, &last.desc, &mut matches_vec, 2, &self.mask, false)?;

            Extractor::filter_matches(&self.kps, last, &mut matches_vec, &mut matches)?;

            println!("{:?} matches", matches.len());
            last.kps = self.kps.clone();
            last.desc = self.orb_desc.clone();
        } else {
            //declare last if last is none
            self.last = Some(LastData {kps: self.kps.clone(), desc: self.orb_desc.clone()});
        }
        Ok(matches)
    }

    pub fn check_last(&self) -> Option<()> {
        match &self.last {
            Some(_) => {
                Some(())
            }
            None => None
        }
    }

    //filter bad matches && converts to points
    fn filter_matches(kps: &Vector<KeyPoint>, last: &mut LastData, matches_vec: &mut Vector<Vector<DMatch>>, matches: &mut Vec<(Point, Point)>) -> opencv::Result<()> {

        if matches_vec.len() == 0 {
            return Ok(());
        }

        *matches = matches_vec 
            .iter()
            .filter(|m1| {
                //filter bad matches using distance
                m1.get(0).unwrap().distance < m1.get(1).unwrap().distance * 0.60
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
            .collect::<Vec<(Point, Point)>>();

        //RANSAC and matrix transform
        let p1: Vector<Point> = Vector::from_iter(matches.iter().map(|(u, _)| *u));
        let p2: Vector<Point> = Vector::from_iter(matches.iter().map(|(_, v)| *v));
        let mut inliners = Mat::default();
        let _ = calib3d::find_fundamental_mat(&p1, &p2, calib3d::RANSAC, 2.0, 0.999, 100, &mut inliners)?;
        let inliners = inliners.to_vec_2d::<u8>()?
            .concat()
            .iter().
            map(|v| *v!= 0)
            .collect::<Vec<_>>();

        let p1: Vec<Point> = p1.iter().zip(&inliners).filter(|(_, x) | **x).map(|(u, _)| u).collect();
        let p2: Vec<Point> = p2.iter().zip(&inliners).filter(|(_, x) | **x).map(|(u, _)| u).collect();
        
        *matches = p1.into_iter().zip(p2.into_iter()).collect();

        Ok(())
    }

}


