import numpy as np
import os
import json

from classifier.config import TARGET_CHANNELS, PHONEME_COUNT, MIN_PHONEME_ID


def save_morlet(filename, morlet):
    """
    Write numpy morlet data to file
    """

    np.save(filename, morlet)


def read_morlet(filename):
    """
    Read numpy morlet data from file
    """

    if os.path.exists(filename):
        return np.load(filename)


def auto_save_morlet(directory, person, phoneme, channel, sample, phoneme_data):
    """
    Write phoneme data to file with the following parameters:

    directory - directory to place files in

    person - index of person / edf file

    phoneme - index of phoneme

    channel - index of used channel

    sample - index of this phoneme's sample

    phoneme_data - phoneme morlet data
    """

    os.makedirs(directory, exist_ok=True)

    save_morlet(f'{directory}/morlet_{person}_{phoneme}_{channel}_{sample}.npy', phoneme_data)


def auto_load_morlet(directory, person, phoneme, channel, sample):
    """
    Read morlet numpy data in the same way as auto_save_morlet()
    """

    return read_morlet(f'{directory}/morlet_{person}_{phoneme}_{channel}_{sample}.npy')


def get_morlet_count(directory, person, phoneme):
    """
    Get count of morlet samples for given person and phoneme ID
    """

    try:
        with open(f'{directory}/count.json', 'r') as f:
            data = json.load(f)

            return data[f'person_{person}'][f'phoneme_{phoneme}']
    except Exception:
        return 0


def set_morlet_count(directory, person, phoneme, count):
    """
    Get count of morlet samples for given person and phoneme ID
    """

    os.makedirs(directory, exist_ok=True)

    try:
        with open(f'{directory}/count.json', 'r') as f:
            data = json.load(f)
    except Exception:
        data = {}

    if f'person_{person}' not in data:
        data[f'person_{person}'] = {}

    data[f'person_{person}'][f'phoneme_{phoneme}'] = count

    with open(f'{directory}/count.json', 'w') as f:
        json.dump(data, f)


def get_total_morlet_count(directory, person):
    """
    Get count of morlet samples for given person ID
    """

    try:
        with open(f'{directory}/count.json', 'r') as f:
            data = json.load(f)

            return data[f'person_{person}']['total']
    except Exception:
        return 0


def set_total_morlet_count(directory, person, count):
    """
    Get count of morlet samples for given person ID
    """

    os.makedirs(directory, exist_ok=True)

    try:
        with open(f'{directory}/count.json', 'r') as f:
            data = json.load(f)
    except Exception:
        data = {}

    if f'person_{person}' not in data:
        data[f'person_{person}'] = {}

    data[f'person_{person}']['total'] = count

    with open(f'{directory}/count.json', 'w') as f:
        json.dump(data, f)


def normalize_labels(labels):
    """
    Normalize label values.

    Source label values start ffrom MIN_PHONEME_ID, normalization substracts
    MIN_PHONEME_ID from each phoneme ID.
    """

    return [p - MIN_PHONEME_ID for p in labels]


def group_morlet_by_phoneme(normalized_morlet_labels, morlet_list):
    """
    Group morlet data by phoneme ID
    """

    result = [[] for _ in range(PHONEME_COUNT)]
    for label, morlet in zip(normalized_morlet_labels, morlet_list):
        result[label].append(morlet)
    return result


def ungroup_morlet_by_phoneme(grouped_morlet_list):
    """
    Performs reverse operation by concatenating all groups
    """

    result = grouped_morlet_list[0]
    labels = [0] * len(grouped_morlet_list[0])
    for i in range(1, PHONEME_COUNT):
        result = result + grouped_morlet_list[i]
        labels = labels + [i] * len(grouped_morlet_list[i])

    return labels, result


def save_person_grouped_morlet_list(directory, person, grouped_morlet_list):
    """
    Save given morlet data for a person and update total person morlet data count

    grouped_morlet_list is a list containing grouped morlet data for each phoneme ID.

    grouped_morlet_list[0] contains all samples for phoneme 0, e.t.c.
    """

    os.makedirs(directory, exist_ok=True)

    # Update total
    set_total_morlet_count(directory, person, sum([len(gml) for gml in grouped_morlet_list]))

    for phoneme in range(PHONEME_COUNT):

        gml = grouped_morlet_list[phoneme]

        # Update count
        set_morlet_count(directory, person, phoneme, len(gml))

        # Iterate over channels & save morlet data
        for index in range(len(gml)):
            for channel in range(TARGET_CHANNELS):
                auto_save_morlet(directory, person, phoneme, channel, index, gml[index][channel])


def load_person_grouped_morlet_list(directory, person):
    """
    Load data for the gven person
    """

    return [
        [
            [
                auto_load_morlet(directory, person, phoneme, channel, index)
                for channel in range(TARGET_CHANNELS)
            ]
            for index in range(get_morlet_count(directory, person, phoneme))
        ]
        for phoneme in range(PHONEME_COUNT)
    ]
