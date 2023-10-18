use fon::chan::Ch32;
use fon::Audio;

use ndarray::prelude::*;

pub fn resample(waves: Array2<f32>, input_sr: u32, output_sr: u32) -> Array2<f32> {
    let mut outputs = vec![];
    for wave in waves.axis_iter(Axis(0)) {
        let wave = wave.to_vec();
        let wave = Audio::<Ch32, 1>::with_f32_buffer(input_sr, wave);
        let mut wave = Audio::<Ch32, 1>::with_audio(output_sr, &wave);
        let wave: Vec<f32> = wave.as_f32_slice().iter().map(|s| *s).collect();
        let wave = Array1::from_vec(wave);
        outputs.push(wave);
    }
    let l = outputs[0].shape()[0];
    let mut output_arr: Array2<f32> = Array::zeros((waves.shape()[0], l));
    for (i, x) in outputs.iter().enumerate() {
        output_arr.slice_mut(s![i, ..]).assign(&x);
    }
    output_arr
}
