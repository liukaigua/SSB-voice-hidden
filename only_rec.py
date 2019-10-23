
"""
Receive composite signal without voice composite.
Left channel receive composite signal, and extract watermark from it at last.
Right channel does nothing.
"""

import numpy as np
import sounddevice as sd
from tools import MaxMinNorm, watermarkExtraction, save

import warnings
warnings.filterwarnings("ignore") # ignore warnings


receivesig = np.array([])

def callback(indata, outdata, frames, time, status):

    global receivesig

    receive_array = indata[:, 0] # left channel is receive signal
    receivesig = np.append(receivesig, receive_array)

    outdata[:] = np.c_[receive_array, receive_array]


print('Receive composite signal, enter anything then press \"Enter\" to end')

sd.default.device = 'wm8960-soundcard' # default audio device
with sd.Stream(samplerate=8000, blocksize=1024, dtype='float32', channels=2, callback=callback):
    input()


print('Extract watermark signal from received composite signal')
receivesig = MaxMinNorm(receivesig)
watermark_rec = watermarkExtraction(receivesig, cp_ratio=16)
watermark_rec = MaxMinNorm(watermark_rec)

print('Play the extracted composite signal')
sd.play(watermark_rec, 8000)
sd.wait()

print('Save the extracted watermark signal')
save_path = 'wav_file/watermark_rec.wav'
save(watermark_rec, save_path)
