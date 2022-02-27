import sys

import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
class constQT:
    def __init__(self):
        self.spectrogram = None #spectro of audio
        self.kernel = None      #CQT Kernel
        self.fs = None          #Sampling freq
        self.bins = None        #self.bins
        self.freq = None        #range of freq in tuples
    def sparseKernel(self, bins, fs, thresh = 0.01, minFreq = 10):

        maxFreq = np.floor(fs // 2)
        self.bins = bins
        self.fs = fs
        self.freq = (minFreq, maxFreq)
        

        Q = 1/((2**(1/bins))-1)
        K = round(bins * np.log2(maxFreq/minFreq)) #Number of channels
        fftLen = 2**np.ceil(np.log2(abs(Q * fs / minFreq)))
        fftLen = int(fftLen)

        sparKernel= np.zeros((K,fftLen), dtype=complex)

        for ind in reversed(range(K)):
            len = 2 * round(Q * fs / (minFreq * 2**(ind / bins))/2) + 1 #always odd
            padLen = int((fftLen - len + 1) / 2)

            sparKernel[ind][padLen : (padLen + len)]= (np.hamming(len) / len) * np.exp(2 * np.pi * 1j * Q * np.arange(len) / len)

        sparKernel = np.fft.fft(sparKernel, axis=1)
        sparKernel[np.absolute(sparKernel) < thresh] = 0
        sparKernel = scipy.sparse.csr_matrix(sparKernel)
        sparKernel = np.conjugate(sparKernel) / fftLen
        self.kernel = sparKernel

    def analyze(self,audio, tres, freq = None):
        if freq is not None:
            if freq != self.fs:
                print("Unmatched Sampling Frequency with Kernel", file= sys.stderr)
                return
        # Derive the number of time samples per time frame
        step = round(self.fs / tres)

        # Compute the number of time frames
        frames = int(np.floor(len(audio) / step))

        # Get th number of frequency channels and the FFT length
        K, fftLen = np.shape(self.kernel)

        # Zero-pad the signal to center the CQT
        audio = np.pad(
            audio,
            (
                int(np.ceil((fftLen - step) / 2)),
                int(np.floor((fftLen - step) / 2)),
            ),
            "constant",
            constant_values=(0, 0),
        )

        # Initialize the CQT spectrogram
        spectrogram = np.zeros((K, frames))

        # Loop over the time frames
        i = 0
        for j in range(frames):
            # Compute the magnitude CQT using the kernel
            spectrogram[:, j] = np.absolute(
                self.kernel * np.fft.fft(audio[i: i + fftLen])
            )
            i = i + step

        self.spectrogram = spectrogram

    def plot(
            self,
            tres,
            step=1
    ):

        minFreq, maxFreq = self.freq
        # Get the number of frequency channels and time frames
        K, frames = np.shape(self.spectrogram)
        
        # Prepare the tick locations and labels for the x-axis
        xtick_locations = np.arange(
            step * tres,
            frames,
            step * tres,
        )
        xtick_labels = np.arange(
            step, frames / tres, step
        ).astype(int)

        # Prepare the tick locations and labels for the y-axis
        ytick_locations = np.arange(0, K, self.bins)
        ytick_labels = (
                minFreq * pow(2, ytick_locations / self.bins)
        ).astype(int)

        # Display the CQT spectrogram in dB and seconds, and Hz
        plt.imshow(
            20 * np.log10(self.spectrogram), aspect="auto", cmap="jet", origin="lower"
        )
        plt.xticks(ticks=xtick_locations, labels=xtick_labels)
        plt.yticks(ticks=ytick_locations, labels=ytick_labels)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        #https://github.com/zafarrafii/Zaf-Python/blob/master/zaf.py
