use crate::fft::{istft, spectrogram};
use ndarray::prelude::*;
use ort::{
    session::Session, tensor::OrtOwnedTensor, Environment, ExecutionProvider,
    GraphOptimizationLevel, OrtResult, SessionBuilder, Value,
};
use std::path::Path;

fn calculate_amplitude(waves: Array2<f32>) -> Array3<f32> {
    let output_length = waves.shape()[2] / 256 + 1;
    let mut output: Array3<f32> = Array::zeros((waves.shape()[0], 1, output_length));
    for (b, wave) in waves.axis_iter(Axis(0)).enumerate() {
        for (i, chunk) in wave.axis_chunks_iter(Axis(0), 256).enumerate() {
            output.slice_mut(s![b, 1, i]).assign(chunk.mean().unwrap());
        }
    }
    output
}

pub struct VoiceConvertor {
    content_encoder: Session,
    pitch_estimator: Session,
    voice_library: Session,
    decoder: Session,
}

impl VoiceConvertor {
    pub fn load(dir_path: &Path, provider: ExecutionProvider) -> OrtResult<Self> {
        let ce_path = dir_path.join("content_encoder.onnx");
        let pe_path = dir_path.join("pitch_estimator.onnx");
        let vl_path = dir_path.join("voice_library.onnx");
        let dec_path = dir_path.join("decoder.onnx");

        let env = Environment::builder()
            .with_name("voice_conversion")
            .with_execution_providers([provider])
            .build()?
            .into_arc();

        let ce_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(ce_path)?;

        let pe_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(pe_path)?;

        let vl_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(vl_path)?;

        let dec_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(dec_path)?;

        Ok(VoiceConvertor {
            content_encoder: ce_sess,
            pitch_estimator: pe_sess,
            voice_library: vl_sess,
            decoder: dec_sess,
        })
    }
}
