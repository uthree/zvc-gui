mod audio;
mod exec_providers;
mod fft;
mod model;
mod wave_file;

use audio::resample;
use exec_providers::available_providers;
use fft::{istft_without_window, stft_without_window};
use model::VoiceConvertor;
use ort::ExecutionProvider;
use std::path::Path;

fn main() {
    /*
    let provider = ExecutionProvider::CUDA(Default::default());
    let mut model = VoiceConvertor::load(&Path::new("../zvc-dev/onnx/"), provider).unwrap();
    let (header, wf) = wave_file::load(Path::new("test.wav"));
    let wf = resample(wf, header.sampling_rate, 16000);
    let wf = model.convert(wf, 0.0, 0.1).unwrap();
    let wf = resample(wf, 16000, header.sampling_rate);
    wave_file::save(header, wf, Path::new("output.wav"));
    */

    println!("{:?}", available_providers());
}
