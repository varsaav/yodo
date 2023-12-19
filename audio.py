import os
import struct
import warnings
import subprocess
import collections
import numpy as np
import tempfile


def normalize(x: np.ndarray):
    ''' Normalize audio to range [-1, 1] without
        changing the gain of the signal.'''

    x = np.asarray(x)
    try:
        dtype = np.iinfo(x.dtype)
        Max = dtype.max
        Min = dtype.min
    except:
        Max = np.max(x)
        Min = np.min(x)

        # If data is already within the (-1, 1) range
        if (Min >= -1.) and (Max <= 1.):
            return x

        # Otherwise, data is normalized 'simetrically',
        # so that zeros remain unchanged
        else:
            Max = max(abs(Max), abs(Min))
            Min = -Max
            warnings.warn(
                "Minimum and maximum values for this datatype "
                "are unknown. Normalizing gain to 0 dB.", UserWarning)

    x = np.asarray(x, dtype=float)
    x[x < 0] /= np.abs(Min)
    x[x > 0] /= np.abs(Max)
    return x


def read_mp3(path: str):
    temp_wav = next(tempfile._get_candidate_names()) + '.wav'
    subprocess.call(['ffmpeg', '-i', path, temp_wav])
    samplerate, samples = read_wav(temp_wav)
    os.remove(temp_wav)
    return samplerate, samples


def read_wav(path: str):
    return _read(path)


def amplitude_to_db(S, ref=1.0, amin=1e-10):
    return power_to_db(S ** 2, ref, amin)


def power_to_db(S, ref=1.0, amin=1e-10):
    power = np.asarray(S)
    # Turn zeros into amin, to avoid calculating log10(0)
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref))
    return log_spec
    

# Since the original scipy.wavfile implementation doesn't handle 24-bits WAV,
# a customized version was used, based on Joseph Ernest's implementation: 
# https://gist.github.com/josephernest/3f22c5ed5dabf1815f16efa8fa53d476

class WavFileWarning(UserWarning):
    pass
    
_ieee = False
    

def _read_fmt_chunk(fid):
    res = struct.unpack('<ihHIIHH',fid.read(20))
    size, comp, noc, rate, sbytes, ba, bits = res
    if (comp != 1 or size > 16):
        if (comp == 3):
          global _ieee
          _ieee = True     
        else: 
          warnings.warn("Unfamiliar format bytes", WavFileWarning)
        if (size>16):
            fid.read(size-16)
    return size, comp, noc, rate, sbytes, ba, bits


def _read_data_chunk(fid, noc, bits, normalized=False):
    size = struct.unpack('<i',fid.read(4))[0]

    if bits == 8 or bits == 24:
        dtype = 'u1'
        bytes = 1
    else:
        bytes = bits//8
        dtype = '<i%d' % bytes
        
    if bits == 32 and _ieee:
       dtype = 'float32'
        
    data = np.fromfile(fid, dtype=dtype, count=size//bytes)
    
    if bits == 24:
        a = np.empty((len(data) // 3, 4), dtype='u1')
        a[:, :3] = data.reshape((-1, 3))
        a[:, 3:] = (a[:, 3 - 1:3] >> 7) * 255
        data = a.view('<i4').reshape(a.shape[:-1])
    
    if noc > 1:
        data = data.reshape(-1,noc)
        
    if bool(size & 1):  # if odd number of bytes, move 1 byte further (data chunk is word-aligned)
      fid.seek(1,1)    

    if normalized:
        if bits == 8 or bits == 16 or bits == 24: 
            normfactor = 2 ** (bits-1)
        data = np.float32(data) * 1.0 / normfactor

    return data


def _skip_unknown_chunk(fid):
    data = fid.read(4)
    size = struct.unpack('<i', data)[0]
    if bool(size & 1):  # if odd number of bytes, move 1 byte further (data chunk is word-aligned)
      size += 1 
    fid.seek(size, 1)


def _read_riff_chunk(fid):
    str1 = fid.read(4)
    if str1 != b'RIFF':
        raise ValueError("Not a WAV file.")
    fsize = struct.unpack('<I', fid.read(4))[0] + 8
    str2 = fid.read(4)
    if (str2 != b'WAVE'):
        raise ValueError("Not a WAV file.")
    return fsize


def _read(file, readmarkers=False, readmarkerlabels=False, readmarkerslist=False, readloops=False, readpitch=False, normalized=False, forcestereo=False):
    
    if hasattr(file,'read'):
        fid = file
    else:
        fid = open(file, 'rb')

    fsize = _read_riff_chunk(fid)
    noc = 1
    bits = 8
    _markersdict = collections.defaultdict(lambda: {'position': -1, 'label': ''})
    loops = []
    pitch = 0.0
    while (fid.tell() < fsize):
        # read the next chunk
        chunk_id = fid.read(4)
        if chunk_id == b'fmt ':
            size, comp, noc, rate, sbytes, ba, bits = _read_fmt_chunk(fid)
        elif chunk_id == b'data':
            data = _read_data_chunk(fid, noc, bits, normalized)
        elif chunk_id == b'cue ':
            str1 = fid.read(8)
            size, numcue = struct.unpack('<ii',str1)
            for c in range(numcue):
                str1 = fid.read(24)
                id, position, datachunkid, chunkstart, blockstart, sampleoffset = struct.unpack('<iiiiii', str1)
                _markersdict[id]['position'] = position  # needed to match labels and markers

        elif chunk_id == b'LIST':
            str1 = fid.read(8)
            size, type = struct.unpack('<ii', str1)
        elif chunk_id in [b'ICRD', b'IENG', b'ISFT', b'ISTJ']:  # see http://www.pjb.com.au/midi/sfspec21.html#i5
            _skip_unknown_chunk(fid)
        elif chunk_id == b'labl':
            str1 = fid.read(8)
            size, id = struct.unpack('<ii',str1)
            size = size + (size % 2)                 # the size should be even, see WAV specfication, e.g. 16=>16, 23=>24
            label = fid.read(size-4).rstrip('\x00')  # remove the trailing null characters
            _markersdict[id]['label'] = label        # needed to match labels and markers

        elif chunk_id == b'smpl':
            str1 = fid.read(40)
            size, manuf, prod, sampleperiod, midiunitynote, midipitchfraction, smptefmt, smpteoffs, numsampleloops, samplerdata = struct.unpack('<iiiiiIiiii', str1)
            cents = midipitchfraction * 1./(2**32-1)
            pitch = 440. * 2 ** ((midiunitynote + cents - 69.)/12)
            for i in range(numsampleloops):
                str1 = fid.read(24)
                cuepointid, type, start, end, fraction, playcount = struct.unpack('<iiiiii', str1) 
                loops.append([start, end])
        else:
            warnings.warn("Chunk skipped", WavFileWarning)
            _skip_unknown_chunk(fid)
    fid.close()

    if data.ndim == 1 and forcestereo:
        data = np.column_stack((data, data))

    _markerslist = sorted([_markersdict[l] for l in _markersdict], key=lambda k: k['position'])  # sort by position
    _cue = [m['position'] for m in _markerslist]
    _cuelabels = [m['label'] for m in _markerslist]
    
    return (rate, data)
