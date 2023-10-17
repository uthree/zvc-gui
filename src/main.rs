mod fft;
mod model;
mod wave_file;

use model::VoiceConvertor;
use ort::ExecutionProvider;
use std::path::Path;

fn main() {
    let provider = ExecutionProvider::CPU(Default::default());
    let model = VoiceConvertor::load(&Path::new("../zvc-dev/onnx/"), provider).unwrap();
}
