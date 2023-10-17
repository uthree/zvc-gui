mod fft;
mod model;
mod wave_file;

use fft::{istft, stft};
use model::VoiceConvertor;
use ort::ExecutionProvider;
use std::path::Path;

fn main() {
    let provider = ExecutionProvider::CPU(Default::default());
    let mut model = VoiceConvertor::load(&Path::new("../zvc-dev/onnx/"), provider).unwrap();
    let (header, wf) = wave_file::load(Path::new("test.wav"));
    //let wf = istft(stft(wf, 1024, 256), 1024, 256);
    let latent = model.encode(wf).unwrap();
    let wf = model.decode(latent).unwrap();
    wave_file::save(header, wf, Path::new("output.wav"));
}
