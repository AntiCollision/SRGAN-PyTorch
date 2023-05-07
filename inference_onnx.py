import onnxruntime as ort
import numpy as np
import time

def make_option_with_provider():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        # ('TensorrtExecutionProvider', {
        #     'device_id': 0,
        #     'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
        #     'trt_fp16_enable': True,
        #     'trt_engine_cache_enable': True,
        #     'trt_engine_cache_path': './trt-engine/'
        # }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        # ('CPUExecutionProvider', {}),
    ]
    return sess_options, providers

def make_licenseplate(path = './super_resolution.onnx'):
    n,h,w,c = (1,3,512,512)
    opt, provider = make_option_with_provider()
    ort_sess = ort.InferenceSession(path, opt, providers=provider)
    
    x = np.random.rand(n,h,w,c).astype(np.float32)
    ort_inputs = {ort_sess.get_inputs()[0].name: x}
    ort_sess.run([], ort_inputs)
    
    return ort_sess

print( ort.get_device()  )

model = make_licenseplate()

n,h,w,c = (1,3,512,512)
x = np.random.rand(n,h,w,c).astype(np.float32)
ort_inputs = {model.get_inputs()[0].name: x}

while True:
    start = time.time()
    model.run([], ort_inputs)
    end = time.time()
    print((end - start) * 1000.0, " ms")
