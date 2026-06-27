import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, input_shape, cache_file):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.batch_size = input_shape[0]
        self.index = 0
        self.images = [
            os.path.join(calib_dir, f)
            for f in sorted(os.listdir(calib_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        assert len(self.images) > 0, f'No images found in {calib_dir}'
        nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.images):
            return None
        h, w = self.input_shape[2], self.input_shape[3]
        batch = []
        for i in range(self.batch_size):
            img = Image.open(self.images[self.index + i]).convert('RGB').resize((w, h))
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            batch.append(arr)
        self.index += self.batch_size
        data = np.ascontiguousarray(np.stack(batch, axis=0))
        cuda.memcpy_htod(self.device_input, data)
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)