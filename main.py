#!/usr/bin/env python3
# -*- ex4mple -*-

import wave
import contextlib
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

# Ask for the file name
# if the file doesn't exist, ask again
while True:
    file_name = input("Enter the file name: ")
    try:
        wavfile.read(file_name)
        break
    except FileNotFoundError:
        print("File not found, please try again.")

FILE = file_name
SAMPLE_RATE = 44100
LOW_CUT = 200
HIGH_CUT = 600


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def main():
    with contextlib.closing(wave.open(FILE, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        durations = int(duration)

    audio_freq_sample, audio_in = wavfile.read(FILE)
    normalized = np.int16((audio_in / audio_in.max()) * 32767)
    sample_number = SAMPLE_RATE * durations

    yf = rfft(normalized)
    xf = rfftfreq(sample_number, 1 / SAMPLE_RATE)

    # Print the frequency of the signal
    print("The frequency of the signal is: ", xf[np.argmax(np.abs(yf))])

    y = butter_bandpass_filter(normalized, LOW_CUT, HIGH_CUT, SAMPLE_RATE, order=5)
    output = np.int16(y * (32767 / y.max()))
    wavfile.write('out_' + FILE, SAMPLE_RATE, output)


if __name__ == '__main__':
    main()
