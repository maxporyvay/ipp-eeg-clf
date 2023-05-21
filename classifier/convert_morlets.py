from classifier.read_edf_extract_split_sectors import extract_strict_sectors, print_sectors_summary, open_audial_edf, open_visual_edf
from classifier.extract_channels_wavelet_pass import subselect_channels, morlet_wavelet_pass, transpose_morlet_channel_data, abs_morlet_data, split_sectors
from classifier.load_save_morlets import normalize_labels, group_morlet_by_phoneme, save_person_grouped_morlet_list

from classifier.config import VISUAL_SUBPATH, AUDIAL_SUBPATH, INPUT_EDF_LIST, SECTOR_LENGTH, CLEARED_PATH, MORLET_ORIGINAL_SAVE_DIR, SECTOR_LENGTH_STEPS, MORLET_FREQ_STEPS


def convert_morlets():
    for visual in [False, True]:
        # Subdirectory matching oble audial or visual
        edf_subdir = VISUAL_SUBPATH if visual else AUDIAL_SUBPATH
        # Full directory path to input edf
        edf_dir = f'{CLEARED_PATH}/{edf_subdir}'
        # Full directory path to morlet files
        morlet_dir = f'{MORLET_ORIGINAL_SAVE_DIR}/width-{SECTOR_LENGTH_STEPS}_height-{MORLET_FREQ_STEPS}/{edf_subdir}'

        print()
        print(f'Preprocessing data in {edf_dir}')
        print(f'Wriging morlets to {morlet_dir}')

        for person, edf_file in enumerate(INPUT_EDF_LIST):
            print()
            print(f'Processing {edf_file}')

            # Open
            if visual:
                edf = open_visual_edf(f'{edf_file}.edf')
            else:
                edf = open_audial_edf(f'{edf_file}.edf')

            # Select segments
            sectors, invalid_sectors, labels = extract_strict_sectors(edf, SECTOR_LENGTH)
            print_sectors_summary(edf, sectors, invalid_sectors)

            # Morlet transform
            print(f'Applying morlet transform for {edf_file}')
            channels_data = subselect_channels(edf)
            _, lengths, durations, splitted = split_sectors(edf, channels_data, sectors)
            t, freq, morlet = morlet_wavelet_pass(splitted)
            morlet = transpose_morlet_channel_data(morlet)
            morlet = abs_morlet_data(morlet)

            print('t.shape', t.shape)
            print('freq.shape', freq.shape)
            print('morlet.shape', morlet.shape)

            # Save data
            print(f'Saving morlet data for {edf_file}')
            normalized_labels = normalize_labels(labels)
            grouped_morlet_list = group_morlet_by_phoneme(normalized_labels, morlet)
            #save_person_grouped_morlet_list(morlet_dir, person, grouped_morlet_list)
