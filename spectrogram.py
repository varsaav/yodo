# Based on Gregor Monke and Etienne Combrisson's implementations of 
# Fractional and Adaptive Superlets as defined by Moca et al. [1] [2]

# Original code:
# https://gist.github.com/EtienneCmb/f76d8b4aba0088aba6a8c07e397a33c2
# https://github.com/TransylvanianInstituteOfNeuroscience/Superlets/blob/main/python/superlet.py

# Papers:
# [1] Time-frequency super-resolution with superlets (2021)
# [2] Fractional superlets (2020)


try:
    import numpy as np
    import cupy as cp
    GPU = True
except:
    cp = np
    cp.asnumpy = np.asarray
    GPU = False

    
def faslt(signal, orders, superlets, fps):
    '''Returns power spectrogram using superlets with fractional orders'''
    signal = np.atleast_2d(signal).astype(np.float64)
    samples_per_frame = int(superlets.samplerate / fps)
    
    n_freqs = len(superlets.foi())
    n_channels, n_samples = signal.shape
    n_frames = int(n_samples / samples_per_frame)
    spec = cp.zeros((n_channels, n_freqs, n_frames), dtype=cp.float64)
    
    padding = (superlets.max_length + 1) // 2
    buffer, begin, end = _alloc_buffer(signal, padding)

    for i, freq in enumerate(superlets.foi()):
        row = cp.ones((n_channels, n_frames), dtype=cp.float64)
        ord_min, ord_frac = int(orders[0]), orders[i]
        
        for order in range(ord_min, int(np.ceil(ord_frac)) + 1):
            wavelet = cp.asarray(superlets(freq, order)).flatten()
            half_width = (len(wavelet) + 1) // 2
            
            for channel in range(n_channels):
                sub_buffer = buffer[channel, begin - half_width : end + half_width]
                convolution = _conv(sub_buffer, wavelet, samples_per_frame)[0:n_frames]

                # Integer part
                if order <= ord_frac:  
                    row[channel] *= (np.sqrt(2) * cp.abs(convolution))

                # Fractional part
                else:  
                    alpha = ord_frac - np.floor(ord_frac)
                    row[channel] *= (np.sqrt(2) * cp.abs(convolution)) ** alpha

        index = n_freqs - i - 1
        exp =  1 / (ord_frac - ord_min + 1)
        spec[:, index] += row ** exp
    
    spec **= 2  # Amplitude to power
    return spec.get() if GPU else spec

    
def aslt(signal, orders, superlets, fps):
    '''Returns amplitude spectrogram using superlets with integer orders'''
    return faslt(signal, np.round(orders, 0), superlets, fps)


class Superlets:
    def __init__(self, samplerate, foi, cycles, orders, mult=True):
        assert foi[0] <= foi[-1]
        assert orders[0] <= orders[-1]
        
        superlets = dict()
        max_length = 0
        
        for i, freq in enumerate(foi):
            wavelets = dict()
            ord_min = int(orders[0])
            ord_max = int(np.ceil(orders[i]))
            
            for order in range(ord_min, ord_max + 1):
                c = (cycles * order) if mult else (cycles + order - 1)
                wavelets[order] = _morlet(freq, c, samplerate)
                max_length = max(max_length, len(wavelets[order]))
                
            superlets[freq] = wavelets

        self.superlets = superlets
        self.samplerate = samplerate
        self.max_length = max_length
    
    def __call__(self, freq, order):
        return self.superlets[freq][order]
    
    def foi(self):
        return self.superlets.keys()


def generate_foi(ref=440, f_min=20, f_max=20000, tones_per_octave=12):
    '''Frequencies of interest according to equal temperament'''
    
    base = 2 ** (1 / tones_per_octave)
    exp_min = np.log(f_min / ref) / np.log(base)
    exp_max = np.log(f_max / ref) / np.log(base)
    
    exponents = np.arange(np.fix(exp_min), np.fix(exp_max) + 1)
    foi = ref * base ** exponents
    return foi


def adaptive_orders(foi, ord_min, ord_max):
    f_min, f_max = foi[0], foi[-1]
    assert f_min < f_max
    return ord_min + (ord_max - ord_min) * (foi - f_min) / (f_max - f_min)


def _morlet(freq, n_cycles, samplerate):
    sd = n_cycles / (5 * freq)
    half = 3 * sd * samplerate
    dt = 1 / samplerate
    t = np.arange(-half, half + 1) * dt

    gauss = 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-(1/2) * (t/sd)**2) * dt
    wavelet = gauss * np.exp(2j * np.pi * freq * t)
    return wavelet


def _conv(arr, kernel, stride):
    sizeof_type = arr.strides[0]
    out_shape = [(len(arr) - len(kernel)) // stride, len(kernel)]
    strides = (stride * sizeof_type, sizeof_type)

    windows = cp.lib.stride_tricks.as_strided(arr, out_shape, strides)
    return cp.tensordot(windows, kernel, axes=(1,0))


def _alloc_buffer(data, padding):
    n_channels, n_samples = np.shape(data)
    length = int(n_samples + 2 * padding)
    buffer = cp.zeros((n_channels, length), dtype=cp.float32)

    begin, end = padding, padding + n_samples
    buffer[:, begin:end] = cp.asarray(data)
    return buffer, begin, end
    