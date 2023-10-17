mod fft;
mod model;
mod wave_file;

use model::VoiceConvertor;
use ort::ExecutionProvider;
use std::path::Path;

fn main() {
    let provider = ExecutionProvider::CPU(Default::default());
    let mut model = VoiceConvertor::load(&Path::new("../zvc-dev/onnx/"), provider).unwrap();
    let (header, wf) = wave_file::load(Path::new("test.wav"));
    println!("ENC");
    let latent = model.encode(wf).unwrap();
    println!("DEC");
    let wf = model.decode(latent, 0.0).unwrap();
    wave_file::save(header, wf, Path::new("output.wav"));
}
