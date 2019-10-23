
"""
Voice composite in real time and receive composite signal at the same time.
Left channel receive composite signal, and extract watermark from it at last.
Right channel records a period of audio as carrier signal, then voice composite in real time.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, hilbert, firwin
from tools import MaxMinNorm, watermarkExtraction, save
import sounddevice as sd

import warnings
warnings.filterwarnings("ignore") # ignore warnings


## Parameters
L = 16 # compression ratio of hidden voice
fc = 2000 # carrier frequency
fs = 8000 # standard sampling rate
TS = 0.02 # unit voice duration of per processing
N = round(TS*fs) # unit voice length of per processing
p = 1 # embedding strength


## Watermark audio
Fs, Sig = wavfile.read('wav_file/watermark.wav')
Sig = MaxMinNorm(Sig.astype(np.int32))

K = Sig.shape[0] // (TS*Fs) # number of unit voice
T = K * TS # total time
voice = Sig[:int(T*Fs), 0].T # left channel

len_watermark = int(len(voice) / Fs*L*fs)
watermark = resample(voice, len_watermark) # resample of watermark


""" Voice composite in real time """

receivesig = np.array([])
sendsig = np.array([])
chunk_num = 0

B = fs / 2 / L
fl = fc / (fs / 2)
fh = (fc + B) / (fs / 2)

CHUNK = N * L
t = np.arange(CHUNK) / fs
f = np.arange(CHUNK) / CHUNK * fs

coefstop = firwin(N + 1, [fl, fh], pass_zero=True) # band elimination filter


print("Voice composite in real time")

def callback(indata, outdata, frames, time, status):

    global receivesig
    global sendsig
    global watermark
    global chunk_num

    receive_array = indata[:, 0] # left channel is receive signal
    receivesig = np.append(receivesig, receive_array)

    carrier_array = indata[:, 1] # right channel is carrier signal
    # carrier = MaxMinNorm(carrier_array)

    if chunk_num < int(K):
        watermark_chunk = watermark[CHUNK * chunk_num: CHUNK * (chunk_num + 1)]
    else:
        if chunk_num == int(K):
            print("Voice composite completed, enter anything and press \"Enter\" to end")
        watermark_chunk = np.array([0] * CHUNK, dtype=np.float32)
    chunk_num = chunk_num + 1

    ## Watermark signal modulation
    hsig = hilbert(watermark_chunk)
    msig = np.multiply(hsig, np.exp(np.pi*2j*fc*t))
    rsig = msig.real

    ## Carrier signal filtering
    buf = np.zeros(2 * N)
    fsig = np.zeros(CHUNK)
    for k in range(int(CHUNK // N)):
        buf[0:N] = buf[N:2*N]
        buf[N:2*N] = carrier_array[k*N:(k+1)*N]
        for n in range(N):
            fsig[k*N+n] = np.multiply(buf[n:n+N+1], coefstop[::-1]).sum(axis=0)

    ## Embed the watermark signal into the carrier signal
    sendsig_piece = (p * fsig + rsig) / (1 + p)
    # sendsig = np.append(sendsig, sendsig_piece)

    outdata[:] = np.c_[sendsig_piece, sendsig_piece]


# sd.default.device = 'seeed-2mic-voicecard' # default audio device
with sd.Stream(samplerate=8000, blocksize=CHUNK, dtype='float32', channels=2, callback=callback):
    input()


""" Extract watermark signal from composite signal which is received from left channel """

print('Extract watermark signal from received composite signal')
watermark_rec = watermarkExtraction(receivesig, cp_ratio=16)
watermark_rec = MaxMinNorm(watermark_rec)

print('Play the extracted watermark signal')
sd.play(watermark_rec, fs)
sd.wait()

print('Save the extracted watermark signal')
save_path = 'wav_file/watermark_rec.wav'
save(watermark_rec, save_path)

