import numpy as np

from classifier.load_save_morlets import load_person_grouped_morlet_list, ungroup_morlet_by_phoneme
from classifier.config import INPUT_EDF_LIST, SECTOR_LENGTH_STEPS, MORLET_FREQ_STEPS, VISUAL_SUBPATH, AUDIAL_SUBPATH, MORLET_ORIGINAL_SAVE_DIR


def load_dataset(visual=None, person=None, sector_length_steps=None, morlet_freq_steps=None):
    """
    Load data with given options.
    If option is set to None, data  for all values of this optio is loaded.

    Returns labels, morlets in ungrouped mode
    """

    labels, morlets = [], []

    person_list = [person]
    visual_list = [visual]

    if person is None:
        person_list = list(range(len(INPUT_EDF_LIST)))

    if visual is None:
        visual_list = [False, True]

    sector_length_steps = sector_length_steps if sector_length_steps is not None else SECTOR_LENGTH_STEPS
    morlet_freq_steps = morlet_freq_steps if morlet_freq_steps is not None else MORLET_FREQ_STEPS

    for visual in visual_list:
        for person in person_list:
            # Subdirectory matching oble audial or visual
            edf_subdir = VISUAL_SUBPATH if visual else AUDIAL_SUBPATH
            # Full directory path to morlet files
            print(f'Morlets directory: {MORLET_ORIGINAL_SAVE_DIR}')
            morlet_dir = f'{MORLET_ORIGINAL_SAVE_DIR}/width-{sector_length_steps}_height-{morlet_freq_steps}/{edf_subdir}'

            # Load
            loaded_data = load_person_grouped_morlet_list(morlet_dir, person=person)

            loaded_labels, loaded_morlets = ungroup_morlet_by_phoneme(loaded_data)
            # print('loaded labels: ', loaded_labels)

            labels += loaded_labels
            morlets += loaded_morlets

    return np.array(labels), np.array(morlets)


def select_phonemes(labels, morlets, phoneme_pair=None):
    """
    Select given pair of phonemes from given data
    """

    if phoneme_pair is None:
        return labels, morlets

    labels, morlets = labels.copy(), morlets.copy()

    # Sumselect required classes
    if phoneme_pair is not None:
        cond = np.isin(labels, phoneme_pair)
        morlets = morlets[cond]
        labels = labels[cond]

        for i in range(len(labels)):
            if labels[i] == phoneme_pair[0]:
                labels[i] = 0
            else:
                labels[i] = 1

    return labels, morlets


def normalize_morlets(morlets):
    return morlets / (morlets.max() - morlets.min())


def train_test_split(labels, morlets, test_size):
    indices = np.arange(len(morlets))
    np.random.shuffle(indices)
    labels, morlets = labels[indices], morlets[indices]
    train_count = int(len(morlets) * (1.0 - test_size))

    train_labels, train_morlets = labels[0:train_count], morlets[0:train_count]
    test_labels, test_morlets = labels[train_count:-1], morlets[train_count:-1]
    return train_labels, train_morlets, test_labels, test_morlets
