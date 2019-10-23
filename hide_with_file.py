
"""
Read an audio as watermark signal (hidden voice), and another as carrier signal,
then the voices are composited, and finally the watermark signal is extracted from the composite signal
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, hilbert, firwin
from tools import MaxMinNorm, save, play
import time

import warnings
import sys
import os
warnings.filterwarnings("ignore") # ignore warnings
os.close(sys.stderr.fileno()) # ignore ALSA errors


## Parameters
L = 16 # compression ratio of hidden voice
fc = 2000 # carrier frequency
fs = 8000 # standard sampling rate
TS = 0.02 # unit voice duration of per processing
N = round(TS*fs) # unit voice length of per processing
p = 1 # embedding strength


## Watermark audio
print('Read the watermark audio')
Fs, Sig = wavfile.read('wav_file/watermark.wav')
Sig = MaxMinNorm(Sig.astype(np.int32))

K = Sig.shape[0] // (TS*Fs) # number of unit voice
T = K * TS # total time
voice = Sig[:int(T*Fs), 0].T # left channel

len_watermark = int(len(voice)/Fs*L*fs)
watermark = resample(voice, len_watermark)


## Carrier audio
print("Read the carrier audio")
Fs, Sig = wavfile.read('wav_file/carrier.wav') # your carrier audio path
Sig = MaxMinNorm(Sig.astype(np.int32))

if Sig.shape[0] < L*T*Fs:
    raise IOError('The carrier audio is not long enough')

music = Sig[:int(L*T*Fs), 0].T
len_carrier = int(len(music)/Fs*fs)
carrier = resample(music, len_carrier) # resample of carrier


## Voice composite
print("Voice composite")

## Watermark signal modulation
t = np.arange(len_watermark) / fs
f = np.arange(len_watermark) / len_watermark * fs
hsig = hilbert(watermark) # hilbert transform
msig = np.multiply(hsig, np.exp(np.pi * 2j * fc * t))
rsig = msig.real

## Carrier signal filtering
B = fs / 2 / L
fl = fc / (fs / 2)
fh = (fc + B) / (fs / 2)
coefstop = firwin(N+1, [fl,fh], pass_zero=True) # band elimination filter

buf = np.zeros(2*N)
fsig = np.zeros(len_carrier)
for k in range(int(len_carrier//N)):
    buf[0:N] = buf[N:2*N]
    buf[N:2*N] = carrier[k*N:(k+1)*N]
    for n in range(N):
        fsig[k*N+n] = np.multiply(buf[n:n+N+1], coefstop[::-1]).sum(axis=0)

## Embed the watermark signal into the carrier signal
sendsig = (fsig + p * rsig) / (1 + p) # composite signal
sendsig = MaxMinNorm(sendsig).astype(np.float32)


## Extract watermark signal from composite signal
print("Extract watermark signal")
coefpass = firwin(N+1, [fl, fh], pass_zero=False) # band-pass filter
coeflow = firwin(N+1, B/(fs/2)) # low pass filter

buf1 = np.zeros(2*N)
buf2 = np.zeros(2*N)
wsig = np.zeros(len(sendsig))
dsig = np.zeros(len(sendsig))

for k in range(int(len(sendsig)//N)):
    buf1[0:N] = buf1[N:2*N]
    buf1[N:2*N] = sendsig[k*N:(k+1)*N]

    for n in range(N):
        wsig[k*N+n] = np.multiply(buf1[n:n+N+1], coefpass[::-1]).sum(axis=0) * np.cos(np.pi*2*fc*t[k*N+n])

    buf2[0:N] = buf2[N:2*N]
    buf2[N:2*N] = wsig[k*N:(k+1)*N]

    for n in range(N):
        dsig[k*N+n] = np.multiply(buf2[n:n+N+1], coeflow[::-1]).sum(axis=0)

len_watermark_rec = int(len(dsig)/(L*fs)*fs)
watermark_rec = resample(dsig, len_watermark_rec)
watermark_rec = MaxMinNorm(watermark_rec).astype(np.float32)


## Saving & playing the extracted watermark signal
save_path = 'wav_file/watermark_rec.wav'
save(watermark_rec, save_path)

print("Play the extracted watermark signal")
t0 = time.time()
play(watermark_rec, fs)
print('Total time: %.2fs' % (time.time() - t0))

## Saving & playing the composite signal
save_path = 'wav_file/sendsig.wav'
save(sendsig, save_path, sample_rate=fs)

print("Play the composite signal")
t0 = time.time()
play(sendsig, fs)
print('Total time: %.2fs' % (time.time() - t0))
