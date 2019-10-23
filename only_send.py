
"""
Voice composite in real time without receive composite signal.
Left channel does nothing.
Right channel records a period of audio as carrier signal, then voice composite in real time.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, hilbert, firwin
from tools import MaxMinNorm, watermarkExtraction
import sounddevice as sd

import warnings
warnings.filterwarnings("ignore")


## Parameters
L = 16 # compression ratio of hidden voice
fc = 2000 # carrier frequency
fs = 8000 # standard sampling rate
TS = 0.02 # unit voice duration of per processing
N = round(TS*fs) # unit voice length of per processing
p = 1 # embedding strength


## Watermark audio
Fs, Sig = wavfile.read('wav_file/watermark.wav') # your watermark audio path
Sig = MaxMinNorm(Sig.astype(np.int32))

K = Sig.shape[0] // (TS*Fs) # number of unit voice
T = K * TS # total time
voice = Sig[:int(T*Fs), 0].T # left channel

len_watermark = int(len(voice) / Fs*L*fs)
watermark = resample(voice, len_watermark) # resample of watermark


""" Voice composite in real time """

B = fs / 2 / L
fl = fc / (fs / 2)
fh = (fc + B) / (fs / 2)

CHUNK = N * L
t = np.arange(CHUNK) / fs
f = np.arange(CHUNK) / CHUNK * fs

coefstop = firwin(N + 1, [fl, fh], pass_zero=True) # band elimination filter

# sendsig = np.array([])
chunk_num = -1

print("Voice composite in real time")

def callback(indata, outdata, frames, time, status):

    # global sendsig
    global watermark
    global chunk_num

    chunk_num = chunk_num + 1

    carrier_array = indata[:, 1] # right channel is carrier signal
    carrier_array = MaxMinNorm(carrier_array)

    if chunk_num < int(K):
        watermark_chunk = watermark[CHUNK * chunk_num: CHUNK * (chunk_num + 1)]

        ## Watermark signal modulation
        hsig = hilbert(watermark_chunk)
        msig = np.multiply(hsig, np.exp(np.pi * 2j * fc * t))
        rsig = msig.real

        ## Carrier signal filtering
        buf = np.zeros(2 * N)
        fsig = np.zeros(CHUNK)
        for k in range(int(CHUNK // N)):
            buf[0:N] = buf[N:2 * N]
            buf[N:2 * N] = carrier_array[k*N:(k+1)*N]
            for n in range(N):
                fsig[k * N + n] = np.multiply(buf[n:n+N+1], coefstop[::-1]).sum(axis=0)

        ## Embed the watermark signal into the carrier signal
        sendsig_piece = (fsig + p * rsig) / (1 + p)
        # sendsig = np.append(sendsig, sendsig_piece)

        outdata[:] = np.c_[sendsig_piece, sendsig_piece]

    else:
        if chunk_num == int(K):
            print("Voice composite completed, enter anything and press \"Enter\" to end")

        outdata[:] = np.c_[carrier_array, carrier_array]


sd.default.device = 'wm8960-soundcard' # default audio device
with sd.Stream(samplerate=8000, blocksize=CHUNK, dtype='float32', channels=2, callback=callback):
    input()


