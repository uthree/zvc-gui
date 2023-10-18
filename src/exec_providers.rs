use ort::execution_providers::ExecutionProvider;

pub fn available_providers() -> Vec<ExecutionProvider> {
    let mut output = Vec::new();
    output.push(ExecutionProvider::CPU(Default::default()));
    let cuda_provider = ExecutionProvider::CUDA(Default::default());
    if cuda_provider.is_available() {
        output.push(cuda_provider);
    }
    output
}
