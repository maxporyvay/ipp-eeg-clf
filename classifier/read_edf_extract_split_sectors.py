import os
import mne
import numpy as np
import matplotlib.pyplot as plt

from classifier.config import OUT_PATH, VISUAL_SUBPATH, AUDIAL_SUBPATH


### тестируется; invalid labels выводится как []
##def extract_strict_sectors(edf, sector_length = 600 ): # Returns sectors[begin,end] and labels
##    """
##    Extract sectors of the given length using begin and start labels.
##    Sample usage is extracting sectors of length 600 ms (1000Hz).
##    """
##    sectors = []
##    labels = []
##
##    number_of_current_phoneme = None
##    counter = 0
##    silent_speach = False
##
##    METKA = edf['METKA']
##    X = METKA[1]
##    Y = METKA[0].T[:,0]
##
##    for index, (timestamp, value) in enumerate(zip(X, Y)):
##        counter -= 1 
##        if value > 0:
##            value = int(value)
##
##            # Phoneme begin 
##            if value // 10 == 1:
##                counter = sector_length
##                number_of_current_phoneme = value % 10
##                silent_speach = True
##            else:
##                silent_speach = False
##
##        if silent_speach and counter == 0:
##            sectors.append((index - sector_length, index))
##            labels.append(number_of_current_phoneme)
##
##    return sectors, [], labels


def extract_strict_sectors(edf, sector_length): # Returns sectors[begin,end] and labels
    """
    Extract sectors of the given length using begin and start labels.
    Sample usage is extracting sectors of length 600 ms (1000Hz).
    """
    
    sectors = []
    labels = []
    invalid_labels = []
    last_sector_end_index = None
    
    METKA = edf['METKA']
    X = METKA[1]
    Y = METKA[0].T[:,0]

    for index, (timestamp, value) in enumerate(zip(X, Y)):
        if value > 0:              
            if last_sector_end_index is not None and index < last_sector_end_index:
                invalid_labels.append(index)
                continue
            
            # Assume that sector [index : index+sector_length] does not 
            #  intersect with other sector [index2 : index2+sector_length]
            value = int(value)
            
            # Phoneme begin
            if value // 10 == 1:                    
                if index + sector_length > len(X):
                    invalid_labels.append(index)
                    continue
                
                # Append sector from current position to current+sector_length as sector
                sectors.append((index, index + sector_length))
                labels.append(value % 10)
                last_sector_end_index = index + sector_length
            
            # Phoneme end
            elif value // 10 == 2:                    
                # Ignore underflow
                if index - sector_length < 0:
                    invalid_labels.append(index)
                    continue
                
                # Append sector from current position to current+sector_length as sector
                sectors.append((index - sector_length, index))
                labels.append(value % 10)
                last_sector_end_index = index
    
    return sectors, invalid_labels, labels


def print_sectors_summary(edf, sectors, missing_labels):
    """
    Print summary info about given set of sectors
    """

    print('sectors:', len(sectors))
    print('invalid:', len(missing_labels))

    METKA = edf['METKA']
    X = METKA[1]

    diff = np.array([X[b] - X[a] for (a, b) in sectors])

    print('min sector length:', np.min(diff))
    print('max sector length:', np.max(diff))
    print('avg sector length:', np.average(diff))


def plot_labels(edf, missing_labels):
    """
    Plot distribution of valid and invalid labels
    """

    METKA = edf['METKA']
    X = METKA[1]
    Y = METKA[0].T[:, 0]

    plt.rcParams["figure.figsize"] = (25, 5)
    plt.rcParams["font.size"] = 14

    for index in range(len(X)):
        if Y[index] > 0:
            if index in missing_labels:
                plt.scatter(X[index], Y[index], color='red', marker='x')
            else:
                plt.scatter(X[index], Y[index], color='blue', marker='.')

    plt.show()


def plot_sectors(sectors, missing_labels):
    """
    Plot segments on single line, inclusing invalid sectors
    """

    plt.rcParams["figure.figsize"] = (25, 2.5)
    plt.rcParams["font.size"] = 14

    # Plot correct labels
    for index, sector in enumerate(sectors):
        plt.plot(sector, (0, 0), color='blue', marker='|', label='sectors' if index == 0 else None)

    # plot invalid labels
    for index, miss in enumerate(missing_labels):
        plt.scatter(miss, 0, color='red', marker='|', label='invalid sectors' if index == 0 else None)

    plt.legend()
    plt.show()


def list_visual_edf():
    """
    List visual EDF file names
    """

    return os.listdir(f'{OUT_PATH}/{VISUAL_SUBPATH}')


def open_visual_edf(filename):
    """
    Open visual data file and return EDF object
    """

    file = f'{OUT_PATH}/{VISUAL_SUBPATH}/{filename}'

    return mne.io.read_raw_edf(file)


def list_audial_edf():
    """
    List audial EDF file names
    """

    return os.listdir(f'{OUT_PATH}/{AUDIAL_SUBPATH}')


def open_audial_edf(filename):
    """
    Open audial data file and return EDF object
    """

    file = f'{OUT_PATH}/{AUDIAL_SUBPATH}/{filename}'

    return mne.io.read_raw_edf(file)
