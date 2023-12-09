use ndarray::prelude::*;
use ort::{
    session::Session, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult,
    SessionBuilder, Value,
};

use std::path::Path;

pub struct Model {
    content_encoder: Session,
    f0_estimator: Session,
    voice_library: Session,
    feature_extractor: Session,
    harmonic_oscillator: Session,
    filter: Session,
}

impl Model {
    pub fn load(dir_path: &Path, provider: ExecutionProvider) -> OrtResult<Self> {
        let ce_path = dir_path.join("content_encoder.onnx");
        let f0e_path = dir_path.join("f0_estimator.onnx");
        let vl_path = dir_path.join("voice_library.onnx");
        let fe_path = dir_path.join("feature_extractor.onnx");
        let ho_path = dir_path.join("harmonic_oscillator.onnx");
        let flt_path = dir_path.join("filter.onnx");

        let env = Environment::builder()
            .with_name("voice_conversion")
            .with_execution_providers([provider])
            .build()?
            .into_arc();

        let ce_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(ce_path)?;

        let f0e_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(f0e_path)?;

        let vl_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(vl_path)?;

        let fe_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(fe_path)?;

        let ho_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(ho_path)?;

        let flt_sess = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(flt_path)?;

        Ok(Model {
            content_encoder: ce_sess,
            f0_estimator: f0e_sess,
            voice_library: vl_sess,
            feature_extractor: fe_sess,
            harmonic_oscillator: ho_sess,
            filter: flt_sess,
        })
    }
}
