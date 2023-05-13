use opencv::{core::*, imgproc, calib3d, features2d};
use opencv::features2d:: {
    ORB, ORB_ScoreType, BFMatcher
};
use opencv::prelude::DescriptorMatcherConst;
use rustslam::convert_to_f32;


struct LastData {
    kps: Vector<KeyPoint>,
    desc: Mat,
}

#[allow(non_snake_case)]
pub struct Extractor {
    orb: Box<dyn ORB>,
    matcher: BFMatcher,
    last: Option<LastData>,
    gray: Mat,
    kps: Vector<KeyPoint>,
    mask: Mat,
    orb_desc: Mat,
    K: Mat,
    K_inv: Mat
}

#[allow(non_snake_case)]
impl Extractor {
    pub fn new(K: Mat) -> opencv::Result<Extractor>{
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

        //KEEP NORM TYPE ON NORM HAMMING FOR ORB
        let matcher = BFMatcher::new(NORM_HAMMING, false)?;
        let gray = Mat::default();
        let kps= Vector::default();
        let mask = Mat::default();
        let orb_desc = Mat::default();
        let K_inv = Extractor::get_k_inv(&K)?;

        Ok(Extractor { orb, matcher , last: None, gray, kps, mask, orb_desc, K, K_inv })
   }

   //get k_inv
   fn get_k_inv(K: &Mat) -> opencv::Result<Mat> {
        let mut K_inv = Mat::new_rows_cols_with_default(3, 3, CV_32F, Scalar::all(0.0))?;
        invert(&K, &mut K_inv, DECOMP_LU)?;
        Ok(K_inv)
   }

   //extract matches
    pub fn extract(&mut self, frame: &Mat) -> opencv::Result<Vec<(Point, Point)>> {
        //convert to grayscale and get features
        imgproc::cvt_color(frame, &mut self.gray, imgproc::COLOR_BGR2GRAY, 1)?;
        features2d::fast(&self.gray, &mut self.kps, 20, true)?;

        //computing descriptors 
        self.orb.compute(&self.gray, &mut self.kps, &mut self.orb_desc)?;

        //get matches if last is some
        let mut matches: Vec<(Point, Point)> = vec![];
        if let Some(last) = &mut self.last {
            //get matches
            let mut matches_vec= Vector::default();
            self.matcher.knn_train_match(&self.orb_desc, &last.desc, &mut matches_vec, 2, &self.mask, false)?;

            Extractor::filter_matches(&self.kps, last, &mut matches_vec, &mut matches, &self.K_inv)?;

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

    fn normalize_point(point: &mut Point, K_inv: &Mat) -> opencv::Result<()> {
        let mut out = Mat::default();

        //turn point into homogenous coordinate 3x1 vector
        let point_mat = Mat::from_slice(&[point.x, point.y, 1])?.reshape(1, 3)?.to_owned();
        let point_mat = convert_to_f32(&point_mat)?;

        gemm(K_inv, &point_mat, 1.0, &Mat::default(), 0.0, &mut out, 0)?;
        point.x = *out.at::<f32>(0)? as i32;
        point.y = *out.at::<f32>(1)? as i32;
        Ok(())
    }

    pub fn denormalize_point(&self, point: &mut Point_<i32>) -> opencv::Result<()> {
        let mut out = Mat::default();

        //turn point into homogenous coordinate 3x1 vector
        let point_mat = Mat::from_slice(&[point.x, point.y, 1])?.reshape(1, 3)?.to_owned();
        let point_mat = convert_to_f32(&point_mat)?;

        gemm(&self.K, &point_mat, 1.0, &Mat::default(), 0.0, &mut out, 0)?;
        point.x = *out.at::<f32>(0)? as i32;
        point.y = *out.at::<f32>(1)? as i32;
        Ok(())
    }

    //filter bad matches && converts to points
    fn filter_matches(kps: &Vector<KeyPoint> ,last: &mut LastData, matches_vec: &mut Vector<Vector<DMatch>>, matches: &mut Vec<(Point, Point)>, K_inv: &Mat) -> opencv::Result<()> {

        *matches = matches_vec 
            .iter()
            .filter(|m1| {
                //filter bad matches using distance
                m1.get(0).unwrap().distance < m1.get(1).unwrap().distance * 0.75
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
        let p1: Vector<Point> = Vector::from_iter(matches.iter().map(|(mut u, _)| {
            Extractor::normalize_point(&mut u, K_inv).unwrap();
            u
        }));
        let p2: Vector<Point> = Vector::from_iter(matches.iter().map(|(_, mut v)| {
            Extractor::normalize_point(&mut v, K_inv).unwrap();
            v
        }));
        let mut inliners = Mat::default();
        let _ = calib3d::find_fundamental_mat(&p1, &p2, calib3d::RANSAC, 2.0, 0.999, 100, &mut inliners)?;

        if inliners.rows() == 0 {
            return Ok(())
        }

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


