import numpy as np
from scipy.signal import resample, firwin
import wave
import struct
import pyaudio


def MaxMinNorm(arr):

    """
    Normalize to 0 ~ 1
    param arr: array to be normalized
    return: The normalized array
    """

    arr_max, arr_min = arr.max(), arr.min()
    return (arr - arr_min) / (arr_max - arr_min)


def watermarkExtraction(com_sig, cp_ratio, fc=2000, fs=8000):

    """
    Extract watermark signal from the composite signal
    param com_sig: composite signal
    param cp_ratio: compression ratio, such as 8, 16 etc
    param fc: carrier frequency
    param fs: standard sampling rate
    return watermark_extraction: extracted watermark signal
    """

    TS = 0.02 # unit voice duration of per processing
    N = round(TS * fs) # unit voice length of per processing
    B = fs / 2 / cp_ratio
    fl = fc / (fs / 2)
    fh = (fc + B) / (fs / 2)

    coefpass = firwin(N + 1, [fl, fh], pass_zero=False)
    coeflow = firwin(N + 1, B / (fs / 2))

    len_com_sig = len(com_sig)

    buf1 = np.zeros(2 * N)
    buf2 = np.zeros(2 * N)
    wsig = np.zeros(len_com_sig)
    dsig = np.zeros(len_com_sig)

    t = np.arange(len_com_sig) / fs
    for k in range(int(len_com_sig // N)):
        buf1[0:N] = buf1[N:2*N]
        buf1[N:2*N] = com_sig[k*N:(k+1)*N]

        for n in range(N):
            wsig[k*N+n] = np.multiply(buf1[n:n+N+1], coefpass[::-1]).sum(axis=0) * np.cos(np.pi*2*fc*t[k*N+n])

        buf2[0:N] = buf2[N:2*N]
        buf2[N:2*N] = wsig[k*N:(k+1)*N]

        for n in range(N):
            dsig[k*N+n] = np.multiply(buf2[n:n+N+1], coeflow[::-1]).sum(axis=0)

    len_watermark_extraction = int(len(com_sig) / (cp_ratio * fs) * fs)
    watermark_extraction = resample(dsig, len_watermark_extraction)

    return watermark_extraction


def save(sig, save_path, sample_rate=8000):

    """
    Save as .wav file with type of int16
    If you want to save with type of float32, annotate the first lines,
        sig = (sig * 65535 - 32768).astype(np.int16)
    and change next lines as follow:
        sig_bytes = struct.pack(str(len(sig)) + 'f', *sig)
        wf.setsampwidth(4)
    """

    sig = (sig * 65535 - 32768).astype(np.int16) # float32 -> int16
    sig_bytes = struct.pack(str(len(sig)) + 'h', *sig)

    wf = wave.open(save_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)

    wf.writeframes(sig_bytes)
    wf.close()


def play(signal, fs):

    """
    Play the audio signal with Pyaudio method
    param signal: audio signal with type of numpy.array
    param fs: sampling rate
    """

    length = len(signal)
    stream_data = struct.pack(str(length) + 'f', *signal)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

    stream.write(stream_data)
    stream.stop_stream()
    stream.close()
    p.terminate()