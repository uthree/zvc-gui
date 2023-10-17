use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::{ndfft_r2c, ndifft_r2c, Complex, FftHandler, R2cFftHandler};
use plotters::prelude::*;
use plotters_backend::BackendColor;
use std::f32::consts::PI;

fn hann_window(size: usize) -> Array1<f32> {
    let window: Array1<f32> = Array::linspace(0.0, 1.0, size);
    let window = (window * 2.0 * PI).mapv(f32::cos) * -0.5 + 0.5 + 1e-6;
    window
}

// Input: [Batch, Length], Output: [Batch, Frequency, Length]
pub fn stft(input: Array2<f32>, n_fft: usize, hop_length: usize) -> Array3<Complex<f32>> {
    // Calculate FFT bin
    let fft_bin = n_fft / 2 + 1;

    // Initialize output
    let mut output = Array::zeros((input.shape()[0], fft_bin, input.shape()[1] / hop_length + 1));

    // Initialize window
    let window = hann_window(n_fft);

    // Initialize FFT Handler
    let mut handler = R2cFftHandler::<f32>::new(n_fft);

    for (b, wave) in input.axis_iter(Axis(0)).enumerate() {
        // Pad wave
        let left_pad_length = n_fft / 2;
        let right_pad_length = n_fft / 2 + (hop_length - (wave.shape()[0] % hop_length));
        let left_pad = Array::zeros(left_pad_length);
        let right_pad = Array::zeros(right_pad_length);
        let wave = concatenate![Axis(0), concatenate![Axis(0), left_pad, wave], right_pad];

        let mut begin = 0;
        let mut end = n_fft;

        let mut i = 0;
        while i < output.shape()[2] {
            // Get window
            let x = wave.slice(s![begin..end]).to_owned() * window.clone();

            // Process FFT
            let mut x_hat = Array::zeros(fft_bin).into_dimensionality().unwrap();
            ndfft_r2c(&x, &mut x_hat, &mut handler, 0);

            // Write output
            output.slice_mut(s![b, .., i]).assign(&x_hat);

            // Move window
            begin += hop_length;
            end += hop_length;
            i += 1;
        }
    }
    output
}

// Input: [Batch, Frequency, Length], Output: [Batch, Length]
pub fn istft(input: Array3<Complex<f32>>, n_fft: usize, hop_length: usize) -> Array2<f32> {
    // Calculate FFT bin
    let fft_bin = n_fft / 2 + 1;

    // Initialize window
    let window = hann_window(n_fft);

    // Calculate wave length
    let wave_length = hop_length * input.shape()[2] + hop_length + n_fft;

    // Initialize output
    let mut output: Array2<f32> = Array::zeros((input.shape()[0], wave_length));

    // Initialize IFFT Handler
    let mut handler = R2cFftHandler::<f32>::new(n_fft);

    for (b, x_hat) in input.axis_iter(Axis(0)).enumerate() {
        let mut begin = 0;
        let mut end = n_fft;
        let mut i = 0;

        while i < input.shape()[2] {
            // Process IFFT
            let mut x: Array1<f32> = Array::zeros((n_fft,));
            let x_hat = x_hat.slice(s![.., i]).map(|elem| elem.clone());

            ndifft_r2c(&x_hat, &mut x, &mut handler, 0);

            // Apply window
            let x = x / window.clone();

            // Write output
            output.slice_mut(s![b, begin..end]).assign(&x);

            // Move window
            i += 1;
            begin += hop_length;
            end += hop_length;
        }
    }
    output
}

// Calculate spectrogram
pub fn spectrogram(input: Array2<f32>, n_fft: usize, hop_length: usize) -> Array3<f32> {
    let complex_spec = stft(input, n_fft, hop_length);
    let spec = complex_spec.map(|x| (x.re * x.re + x.im * x.im).sqrt());
    spec
}

// For debugging
pub fn plot_spectrogram(spec: Array3<f32>, file_path: &str) {
    // Initializze backend
    let mut backend =
        BitMapBackend::new(file_path, (spec.shape()[2] as u32, spec.shape()[1] as u32));
    // Draw spectrogram
    for x in 0..spec.shape()[2] - 1 {
        for y in 0..spec.shape()[1] - 1 {
            let m = (spec[[0, y, x]] * 256.0) as u8;
            let color = BackendColor {
                alpha: 1.0,
                rgb: (m, m, m),
            };
            backend
                .draw_rect(
                    (x as i32, y as i32),
                    (x as i32 + 1, y as i32 + 1),
                    &color,
                    true,
                )
                .unwrap()
        }
    }
    // Save file
    backend.present().unwrap();
}
