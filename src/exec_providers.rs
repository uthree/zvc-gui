use ort::execution_providers::{CUDAExecutionProviderOptions, ExecutionProvider};

pub fn available_providers() -> Vec<ExecutionProvider> {
    let mut output = Vec::new();
    output.push(ExecutionProvider::CPU(Default::default()));
    let mut id: u32 = 0;
    loop {
        let mut opt: CUDAExecutionProviderOptions = Default::default();
        opt.device_id = id;
        let mut prov = ExecutionProvider::CUDA(opt);
        match prov {
            ExecutionProvider::CUDA(ref mut cuda_opt) => {
                cuda_opt.device_id = id;
            }
            _ => unreachable!(),
        }
        if prov.is_available() {
            output.push(prov);
        } else {
            break;
        }
        id += 1;
    }
    output
}
