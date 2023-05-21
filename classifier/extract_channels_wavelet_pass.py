import mne
import numpy as np
import scipy.signal

from classifier.config import TARGET_CHANNELS, TARGET_CHANNEL_SETS, SOURCE_FREQ, LOW_PASS_FREQ, HIGH_PASS_FREQ, SECTOR_LENGTH_STEPS, SECTOR_LENGTH, MAX_MORLET_FREQ, MORLET_FREQ_STEPS


def subselect_channels(edf):
    print(f'Available channels: {edf.ch_names}')

    channels = [None] * TARGET_CHANNELS
    for i in range(TARGET_CHANNELS):
        # Iterate over all channels find compatible channel names
        for compatible in range(len(TARGET_CHANNEL_SETS[i])):
            try:
                channels[i] = edf[TARGET_CHANNEL_SETS[i][compatible]][0][0]
                break
            except Exception:
                continue

        if channels[i] is None:
            raise RuntimeError(f'No compatible channels found for channels {TARGET_CHANNEL_SETS[i]}')

    return channels


def butterworth_filter_pass(edf, channels_data):
    filtered = [None] * TARGET_CHANNELS

    for index, cd in enumerate(channels_data):
        filtered[index] = mne.filter.filter_data(cd, SOURCE_FREQ, LOW_PASS_FREQ, HIGH_PASS_FREQ, method='iir')

    return filtered


def split_sectors(edf, channels_data, sectors):
    """
    Performs slicing of the given channels using sector info data.
    Returns label number, split length, split duration and splitted data for channels
    """

    METKA = edf['METKA']
    X = METKA[1]
    Y = METKA[0][0]  # .T[:,0]

    splitted  = [[None] * len(sectors) for i in range(len(channels_data))]
    lengths   = [None] * len(sectors)
    durations = [None] * len(sectors)
    labels    = [None] * len(sectors)

    for index in range(len(sectors)):
        (a, b) = sectors[index]

        labels[index]    = int(Y[a]) % 10
        lengths[index]   = b - a
        durations[index] = X[b] - X[a]

        for index2, f in enumerate(channels_data):
            splitted[index2][index] = f[a:b]

    return labels, lengths, durations, splitted


def single_morlet_wavelet_pass(sample, w=6.0):
    """
    Apply wavelet transform on the givven sample
    """

    t, dt = np.linspace(0, SECTOR_LENGTH / SOURCE_FREQ, SECTOR_LENGTH, retstep=True)
    freq = np.linspace(1, MAX_MORLET_FREQ, MAX_MORLET_FREQ)
    fs = 1 / dt
    widths = w * fs / (2 * freq * np.pi)

    return t[::SECTOR_LENGTH // SECTOR_LENGTH_STEPS], freq[::MAX_MORLET_FREQ // MORLET_FREQ_STEPS], scipy.signal.cwt(sample, scipy.signal.morlet2, widths, w=w)[::MAX_MORLET_FREQ // MORLET_FREQ_STEPS, ::SECTOR_LENGTH // SECTOR_LENGTH_STEPS]


def rescale_morlet_plz(sample):
    """
    Rescale from shape (MAX_MORLET_FREQ, SECTOR_LENGTH) to shape
    (MORLET_FREQ_STEPS, SECTOR_LENGTH_STEPS)
    """

    NW = SECTOR_LENGTH_STEPS
    FW = (SECTOR_LENGTH // SECTOR_LENGTH_STEPS)

    if SECTOR_LENGTH == SECTOR_LENGTH_STEPS:
        return sample

    sample = np.reshape(sample, (MAX_MORLET_FREQ, NW, FW)).mean(axis=2)

    if MAX_MORLET_FREQ != MORLET_FREQ_STEPS:
        raise RuntimeError('Incomplete code')

    return sample


def morlet_wavelet_pass(channel_splitted_data, w=6.0):
    """
    Performs wavelet transform over the given data. Returns 2D matrixes
    representing morlet transform application result for each of 4 channels for
    each of N samples.

    channel_splitted_data contains 4 channels, each has a set of splitted
    samples in it.
    """

    t, dt = np.linspace(0, SECTOR_LENGTH / SOURCE_FREQ, SECTOR_LENGTH, retstep=True)
    freq = np.linspace(1, MAX_MORLET_FREQ, MAX_MORLET_FREQ)
    fs = 1 / dt
    widths = w * fs / (2 * freq * np.pi)

    FW = (MAX_MORLET_FREQ // MORLET_FREQ_STEPS)
    FH = (SECTOR_LENGTH // SECTOR_LENGTH_STEPS)

    return t[::FH], freq[::FW], [
        [
            rescale_morlet_plz(scipy.signal.cwt(channel_splitted_data[channel][index], scipy.signal.morlet2, widths, w=w))
            for index in range(len(channel_splitted_data[channel]))
        ]
        for channel in range(TARGET_CHANNELS)
    ]


def transpose_morlet_channel_data(morlet_channel_data):
    """
    Perform transposition of channel data so order changes from

    morlet_channel_data[channel][index]

    to

    morlet_channel_data[index][channel]
    """

    return [
        [
            morlet_channel_data[channel][index]
            for channel in range(TARGET_CHANNELS)
        ]
        for index in range(len(morlet_channel_data[0]))
    ]


def abs_morlet_data(morlet):
    return np.abs(morlet)
