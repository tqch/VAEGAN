from functools import reduce
import operator as op
import numpy as np


class SNR:
    def __init__(self):
        self.running_snr = None
        self.count = 0

    def reset(self):
        self.running_snr = 0
        self.count = 0

    def accumulate(self, x):
        # cxhxw = reduce(op.mul, x.shape[1:], 1)
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        mse = ((x - mean) ** 2).mean(axis=(1, 2, 3))
        mean = mean[0]
        count = x.shape[0]
        snr = 10 * np.log10(mean**2 / mse).mean()
        if self.count == 0:
            self.running_snr = snr
        else:
            alpha = count / (self.count + count)
            self.running_snr += alpha * (snr - self.running_snr)
            self.count += count

    def __call__(self, x):
        self.accumulate(x)

    def get_metrics(self):
        return self.running_snr


class PSNR:
    def __init__(self, peak_value=1):
        self.peak_value = peak_value
        self.running_psnr = None
        self.count = 0

    def reset(self):
        self.running_psnr = 0
        self.count = 0

    def accumulate(self, x):
        # cxhxw = reduce(op.mul, x.shape[1:], 1)
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        mse = ((x - mean) ** 2).mean(axis=(1, 2, 3))
        count = x.shape[0]
        psnr = 10 * np.log10(self.peak_value**2 / mse).mean()
        if self.count == 0:
            self.running_psnr = psnr
        else:
            alpha = count / (self.count + count)
            self.running_psnr += alpha * (psnr - self.running_psnr)
            self.count += count

    def __call__(self, x):
        self.accumulate(x)

    def get_metrics(self):
        return self.running_psnr
