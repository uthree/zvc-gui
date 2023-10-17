use crate::fft::{istft, spectrogram};
use ndarray::prelude::*;
use ndrustfft::Complex;
use ort::{
    session::Session, tensor::OrtOwnedTensor, Environment, ExecutionProvider,
    GraphOptimizationLevel, OrtResult, SessionBuilder, Value,
};
use std::ops::Deref;
use std::path::Path;

fn calculate_amplitude(waves: Array2<f32>) -> Array3<f32> {
    let output_length = waves.shape()[1] / 256 + 1;
    let mut output: Array3<f32> = Array::zeros((waves.shape()[0], 1, output_length));
    for (b, wave) in waves.axis_iter(Axis(0)).enumerate() {
        for (i, chunk) in wave.axis_chunks_iter(Axis(0), 256).enumerate() {
            println!("{:?}", i);
            output
                .slice_mut(s![b, 1, i])
                .assign(&arr0(chunk.map(|x| f32::abs(*x)).mean().unwrap()));
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

pub struct LatentRepresentation {
    pub amplitude: Array3<f32>,
    pub content: Array3<f32>,
    pub f0: Array3<f32>,
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

    pub fn encode(&mut self, waves: Array2<f32>) -> OrtResult<LatentRepresentation> {
        let amp = calculate_amplitude(waves.clone());
        let spec = CowArray::from(spectrogram(waves, 1024, 256))
            .into_dimensionality()
            .unwrap();

        let ce_inputs = vec![Value::from_array(self.content_encoder.allocator(), &spec)?];
        let pe_inputs = vec![Value::from_array(self.pitch_estimator.allocator(), &spec)?];

        let con = self.content_encoder.run(ce_inputs)?[0]
            .try_extract()?
            .view()
            .deref()
            .to_owned()
            .into_dimensionality()
            .unwrap();
        let f0 = self.pitch_estimator.run(pe_inputs)?[0]
            .try_extract()?
            .view()
            .deref()
            .to_owned()
            .into_dimensionality()
            .unwrap();

        Ok(LatentRepresentation {
            amplitude: amp,
            content: con,
            f0: f0,
        })
    }

    pub fn decode(&mut self, input: LatentRepresentation, alpha: f32) -> OrtResult<Array2<f32>> {
        let features = CowArray::from(input.content).into_dimensionality().unwrap();
        let vl_inputs = vec![Value::from_array(
            self.voice_library.allocator(),
            &features,
        )?];

        // match vectors
        let features: ArrayD<f32> = self.voice_library.run(vl_inputs)?[0]
            .try_extract()?
            .view()
            .deref()
            .to_owned();

        let features = CowArray::from(features);
        let f0 = CowArray::from(input.f0.into_dimensionality().unwrap());
        let amp = CowArray::from(input.amplitude.into_dimensionality().unwrap());

        let decoder_inputs = vec![
            Value::from_array(self.decoder.allocator(), &features)?,
            Value::from_array(self.decoder.allocator(), &f0)?,
            Value::from_array(self.decoder.allocator(), &amp)?,
        ];

        let decoder_outputs = self.decoder.run(decoder_inputs)?;
        let mag = &decoder_outputs[0];
        let phase = &decoder_outputs[1];

        let mag: Array3<f32> = mag
            .try_extract()?
            .view()
            .deref()
            .to_owned()
            .into_dimensionality()
            .unwrap();
        let phase: Array3<f32> = phase
            .try_extract()?
            .view()
            .deref()
            .to_owned()
            .into_dimensionality()
            .unwrap();

        let mag = mag.map(|x| x.exp());
        let phase = phase.map(|m| Complex {
            re: m.cos(),
            im: m.sin(),
        });
        let s = phase * mag;

        Ok(istft(s, 1024, 256))
    }
}
