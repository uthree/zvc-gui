use ndarray::prelude::*;
use std::fs::File;
use std::path::Path;
use wav;

pub fn load(path: &Path) -> (wav::Header, Array2<f32>) {
    let mut inp_file = File::open(path).unwrap();
    let (header, data) = wav::read(&mut inp_file).unwrap();

    let data = match data {
        wav::BitDepth::Eight(_) => Array::from_vec(
            data.as_eight()
                .unwrap()
                .iter()
                .map(|x| (*x as f32) / 128.0)
                .collect(),
        ),

        wav::BitDepth::TwentyFour(_) => Array::from_vec(
            data.as_twenty_four()
                .unwrap()
                .iter()
                .map(|x| (*x as f32) / 8388608.0)
                .collect(),
        ),
        wav::BitDepth::ThirtyTwoFloat(d) => Array::from_vec(d),
        wav::BitDepth::Sixteen(_) => Array::from_vec(
            data.as_sixteen()
                .unwrap()
                .iter()
                .map(|x| (*x as f32) / 32768.0)
                .collect(),
        ),

        wav::BitDepth::Empty => panic!("Wave file is empty"),
    }
    .insert_axis(Axis(0));
    (header, data)
}

pub fn save(data: Array2<f32>, header: wav::Header, path: &Path) {
    let data = data.remove_axis(Axis(0));
    let data = wav::BitDepth::Sixteen(
        data.to_vec()
            .iter()
            .map(|x| (*x * 32768.0) as i16)
            .collect(),
    );
    let mut out_file = File::create(path).unwrap();
    wav::write(header, &data, &mut out_file).unwrap();
}
