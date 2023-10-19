mod audio;
mod exec_providers;
mod fft;
mod model;
mod wave_file;

use audio::resample;
use eframe::egui::{self, ComboBox};
use exec_providers::available_providers;
use model::VoiceConvertor;
use ndarray::prelude::*;
use once_cell::sync::Lazy;
use ort::ExecutionProvider;
use std::borrow::BorrowMut;
use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::Path;
use std::sync::Mutex;

static model: Lazy<Mutex<Option<VoiceConvertor>>> = Lazy::new(|| Mutex::new(None));
static pitch_shift: Lazy<Mutex<f32>> = Lazy::new(|| Mutex::new(0.0));
static alpha: Lazy<Mutex<f32>> = Lazy::new(|| Mutex::new(0.0));

fn update_model_paths(model_paths: &mut Vec<OsString>) {
    model_paths.clear();
    for path in fs::read_dir("./models/").unwrap() {
        model_paths.push(path.unwrap().path().as_os_str().to_os_string());
    }
}

fn main() -> eframe::Result<()> {
    // initialize model directory
    if !Path::new("./models/").exists() {
        std::fs::create_dir(Path::new("./models/")).unwrap();
    }

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(320.0, 240.0)),
        ..Default::default()
    };

    let mut available_model_paths: Vec<OsString> = Vec::new();
    let mut model_path = OsString::from("Select Model");
    let mut pitch_shift_local: f32 = 0.0;
    let mut alpha_local: f32 = 0.0;
    let providers = available_providers();
    let mut provider_id = 0;

    update_model_paths(&mut available_model_paths);

    eframe::run_simple_native("Voice Conversion", options, move |ctx, _frame| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Execusion Provider: ");
                egui::ComboBox::from_label("Provider")
                    .selected_text(providers[provider_id].as_str())
                    .show_ui(ui, |ui| {
                        for (id, prov) in providers.iter().enumerate() {
                            ui.selectable_value(&mut provider_id, id, prov.as_str());
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Model: ");
                if ui.button("Load").clicked() {
                    update_model_paths(&mut available_model_paths);
                    let load_result = VoiceConvertor::load(
                        Path::new(&model_path),
                        providers[provider_id].clone(),
                    );
                    if load_result.is_ok() {
                        let loaded = load_result.unwrap();
                        *model.lock().unwrap() = Some(loaded).into();
                    }
                }

                egui::ComboBox::from_label("Model")
                    .selected_text(model_path.to_str().unwrap())
                    .show_ui(ui, |ui| {
                        for p in available_model_paths.iter() {
                            ui.selectable_value(&mut model_path, p.clone(), p.to_str().unwrap());
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Pitch Shift: ");
                ui.add(egui::Slider::new(&mut pitch_shift_local, -24.0..=24.0));
            });

            ui.horizontal(|ui| {
                ui.label("Alpha: ");
                ui.add(egui::Slider::new(&mut alpha_local, 0.0..=1.0));
            });

            *alpha.lock().unwrap() = alpha_local;
            *pitch_shift.lock().unwrap() = pitch_shift_local;
        });
    })?;

    Ok(())
}
