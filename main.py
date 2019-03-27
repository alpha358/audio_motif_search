import mir_eval.display
from scipy.io import wavfile
from scipy import signal
import IPython.display
import matplotlib.pyplot as plt
import librosa.display
import matplotlib.pyplot as plt  # grafikams
import numpy as np
import time


from core import * # Matrix profile

# Importuojame reikalingus python paketus
# todo: comment
import scipy.io as sio

# ============================================================================ #
#                                 SPECTROGRAMS                                 #
# ============================================================================ #

sample_rate = 48000
stft_hop = int(sample_rate * 23 / 1000) # 23 ms
stft_window = int(sample_rate * 46 / 1000)  # 46 ms

def get_mel_spectrogram(samples, sample_rate=48000):
    # Furje transformacijos laiko langas.
    stft_window = int(sample_rate * 46 / 1000)  # 46 ms

    # FT laiko lango žingsnis
    stft_hop = int(sample_rate * 23 / 1000)  # 23 ms

    mel_filters = 32

    #  Mel-spectrogram  is
    # extracted with the following parameters, which are commonly
    # used  in  MIR:  46  milliseconds  short  time  Fourier  transform
    # (STFT) window, 23 milliseconds STFT hop, and 32 Mel-scale
    # triangular filters.

    mel = librosa.feature.melspectrogram(
        y=samples[::, 1],
        sr=sample_rate,
        n_fft=stft_window,
        hop_length=stft_hop,
        n_mels=mel_filters
    )
    return mel


def plot_mel_spectrogram(mel, stft_hop=stft_hop, sample_rate=48000):
    plt.figure(figsize=(15, 5))
    eps = np.finfo(np.float32).eps  # machine epsilon

    t = np.linspace(0, stft_hop / sample_rate *
                    mel.shape[1] / 60, mel.shape[1])
    freqs = np.linspace(1, 32, 32)
    # Time, Freqs = np.meshgrid(t, freqs)

    plt.pcolormesh(t, freqs, np.log(mel[:, :] + eps), cmap='jet')

    plt.colorbar()
    # plt.imshow(spectrogram)
    plt.ylabel('Dažnis, s.v.')
    plt.xlabel('Laikas, min.')
    plt.title('Mel spektrograma (log mastelis)')
    plt.show()


# ============================================================================ #
#                                     MISC                                     #
# ============================================================================ #

def get_audio_idx(spectrogram_idx, stft_hop, sample_rate):
    '''
    Purpose: return index of music time series that correspond to specific
             Mel Spectrogram time index.
    '''
    return spectrogram_idx * stft_hop


def seconds_to_steps(t):
    '''
    Return number of Mel spectrogram steps
    for a number of seconds
    '''
    return int(218/5 * t)

def norm(x): return (x - np.mean(x))/np.std(x)

# ============================================================================ #
#                                MATRIX PROFILE                                #
# ============================================================================ #

def get_m_profile(A, B, m):
    '''
    A --- time series 1
    B --- time series 2
    m --- motif length
    '''
    matrix_profile, mpIndex = stamp(A, B, m)
#     matrix_profile, mpIndex = stomp(A, m, tsB=B)
#     MP, I = stamp(ts,query,15)

    return matrix_profile, mpIndex



# ============================================================================ #
#                               FILTER TOP MOTIFS                              #
# ============================================================================ #

def top_motifs(matrix_profile, mpIndex, N, min_distance=0):
    '''
    matrix_profile --- matrix profile (difference measure)
    mpIndex       --- matrix profile index
    N  --- number of motifs
    min_discante --- minimal distance from motifs in time series steps
                    may be 0 for different tracks comparison
    '''
    idx_1_ = np.argsort(matrix_profile) # ascending order
    idx_2_ = mpIndex[idx_1_]

    # lists for selected motifs
    idx_1 = []
    idx_2 = []
    distanceL2 = []

    # select the motifs
    for n in range(N):
        if np.abs(idx_1_[n] - idx_2_[n]) > min_distance:
            # Select only motifs that are enough far away
            idx_1.append( int(idx_1_[n]) )
            idx_2.append( int(idx_2_[n]) )
            distanceL2.append( matrix_profile[idx_1_[n]])


    return idx_1, idx_2, distanceL2



# ============================================================================ #
#                                  PLAY MOTIFS                                 #
# ============================================================================ #


def display_motif(idx, motif_len, samples):
    t1 = idx[0]
    t2 = idx[0] + motif_len

    # time idx in audio file
    T1 = get_audio_idx(t1, stft_hop, sample_rate)
    T2 = get_audio_idx(t2, stft_hop, sample_rate)

    # player
    IPython.display.display(
        IPython.display.Audio(samples[T1:T2, 0], rate=sample_rate)
    )


def play_motifs(idx1, idx2, motif_len, samplesA, samplesB):
    '''
    '''
    display_motif(idx1, motif_len, samplesA)
    display_motif(idx2, motif_len, samplesB)

