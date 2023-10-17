mod fft;

use fft::{istft, plot_spectrogram, spectrogram, stft};
use std::fs::File;
use std::path::Path;
use wav;
use wav::BitDepth;

use ndarray::prelude::*;

fn main() {
    let mut inp_file = File::open(Path::new("test.wav")).unwrap();
    let (header, data) = wav::read(&mut inp_file).unwrap();

    // Load data
    let data = Array::from_vec(
        data.as_sixteen()
            .unwrap()
            .iter()
            .map(|x| (*x as f32) / 32768.0)
            .collect(),
    )
    .insert_axis(Axis(0));

    let x_hat = stft(data, 1024, 256);

    let x = istft(x_hat, 1024, 256);

    let data = x.remove_axis(Axis(0));
    let data = BitDepth::Sixteen(
        data.to_vec()
            .iter()
            .map(|x| (*x * 32768.0) as i16)
            .collect(),
    );
    let mut out_file = File::create(Path::new("output.wav")).unwrap();
    wav::write(header, &data, &mut out_file).unwrap();
}
