LOCALIZED = False

if LOCALIZED:
    DATA_PATH = 'LocalizedM3-cleared'
    TARGET_CHANNELS = 13
    TARGET_CHANNEL_SETS = [
        ['EEG F7-A1', 'EEG F7-M1'],
        ['EEG F3-A1', 'EEG F3-M1'],
        ['EEG T3-A1', 'EEG T3-M1'],
        ['EEG C3-A1', 'EEG C3-M1'],
        ['L El 1 1'],
        ['L El 1 2'],
        ['L El 1 3'],
        ['L El 2 1'],
        ['L El 2 2'],
        ['L El 2 3'],
        ['L El 3 1'],
        ['L El 3 2'],
        ['L El 3 3']
    ]
else:
    DATA_PATH = 'Cleared'
    TARGET_CHANNELS = 4
    TARGET_CHANNEL_SETS = [
        ['EEG F7-A1', 'EEG F7-M1'],
        ['EEG F3-A1', 'EEG F3-M1'],
        ['EEG T3-A1', 'EEG T3-M1'],
        ['EEG C3-A1', 'EEG C3-M1']
    ]

OUT_PATH       = f'{DATA_PATH}-converted'
CLEARED_PATH   = OUT_PATH
VISUAL_SUBPATH = 'Visual'
AUDIAL_SUBPATH = 'Audial'

SOURCE_FREQ         = 1000  # Article: 1000Hz
SECTOR_LENGTH       = 600
SECTOR_LENGTH_STEPS = 600
MAX_MORLET_FREQ     = 30
MORLET_FREQ_STEPS   = 30
LOW_PASS_FREQ       = 3
HIGH_PASS_FREQ      = 30
MAX_SAMPLE_LENGTH   = 1.5

# Phonemes are enumerated in range 2, 3, 4, 5, 6, 7, 8
MIN_PHONEME_ID = 2
PHONEME_COUNT  = 7

# Directories
MORLET_ORIGINAL_SAVE_DIR = 'morlet-original'
if LOCALIZED:
    MORLET_ORIGINAL_SAVE_DIR = 'morlet-original-localized'

# List of EDF files to use
# These files are taken from CLEARED_PATH/VISUAL_SUBPATH and CLEARED_PATH/AUDIAL_SUBPATH
INPUT_EDF_LIST = [
    'Antonova',
    'BazvlkD',
    'DashaPap',
    'Drachenko',
    'Gordokov',
    'Manenkov',
    'PavluhinN',
    'RylkovS',
    'Sazonova',
    'VinickiD',
]

# Flags
CONVERT_MORLETS   = False
SAVE_CHECKPOINTS  = True
SAVE_CONFIG       = True
RUN_TEST          = True


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__  = dict.get
    __setattr__  = dict.__setitem__
    __delattr__  = dict.__delitem__

    def copy(self):
        my_copy = type(self)()
        for k, v in self.items():
            my_copy[k] = v
        return my_copy


test_config = dotdict()

###################################################

# Enable / disable config
BINARY_SINGLE = True

if BINARY_SINGLE:
    test_config = dotdict()

    # Data selection
    # Whiat data to use (visual, audial, all together)
    test_config.visual          = None

    # Set to True to enable binary classification mode
    test_config.binary          = True

    if test_config.binary:

        # What person to use to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = None

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = False

        # What phonemes to use during train
        # Set to None to train on each pair combination
        # Set to pair to train on this data pair
        test_config.phoneme_classes = None

    else:

        # List of persons to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = 1

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = True

        # Phoneme classes to use during train
        # Set to None to use all phonemes
        test_config.phoneme_classes = None

    # Train properties
    test_config.test_size         = 0.2
#    test_config.batch_size        = 4
    test_config.batch_size        = 8
    test_config.epochs            = 60
    test_config.use_conv_sigmoid  = False
    test_config.use_dense_sigmoid = False  # test_config.binary

    # Train autoconfig
    test_config.lr_step_size     = 30
#    test_config.lr_step_size     = 40
    test_config.lr_start         = 0.01

    # Transform properties
    test_config.shift_transform_scale = 1.1
    test_config.shift_transform_roll  = 1.0
    test_config.noise_transform_scale = 0.002
    test_config.make_flip_along_time  = False

    test_config.seed       = 42

    # Model properties
    if not LOCALIZED:
        test_config.conv_layers = (
            {
                'out': 6,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 12,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 64,
                'dropout': 0
            },
        )
    else:
        test_config.conv_layers = (
            {
                'out': 20,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 40,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 64,
                'dropout': 0
            },
        )

###################################################

# Enable / disable config
BINARY_ALL = False

if BINARY_ALL:
    test_config = dotdict()

    # Data selection
    # Whiat data to use (visual, audial, all together)
    test_config.visual          = None

    # Set to True to enable binary classification mode
    test_config.binary          = True

    if test_config.binary:

        # What person to use to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = 1

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = True

        # What phonemes to use during train
        # Set to None to train on each pair combination
        # Set to pair to train on this data pair
        test_config.phoneme_classes = None

    else:

        # List of persons to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = 1

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = False

        # Phoneme classes to use during train
        # Set to None to use all phonemes
        test_config.phoneme_classes = None

    # Force enable/disable dataset reloading, for example, when performing multiple restarts
    RELOAD_DATASET = False

    # Train properties
    test_config.test_size         = 0.2
    test_config.batch_size        = 8
    test_config.epochs            = 50
    test_config.use_conv_sigmoid  = False
    test_config.use_dense_sigmoid = False  # test_config.binary

    # Train autoconfig
    test_config.lr_step_size     = 40
    test_config.lr_start         = 0.01

    # Transform properties
    test_config.shift_transform_scale = 1.1
    test_config.shift_transform_roll  = 1.0
    test_config.noise_transform_scale = 0.002
    test_config.make_flip_along_time  = False

    test_config.seed       = 42

    # Model properties
    if not LOCALIZED:
        test_config.conv_layers = (
            {
                'out': 8,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 16,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 128,
                'dropout': 0
            },
        )
    else:
        test_config.conv_layers = (
            {
                'out': 26,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 52,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 128,
                'dropout': 0
            },
        )

###################################################

# Enable / disable config
MULTICLASS_SINGLE = False

if MULTICLASS_SINGLE:
    test_config = dotdict()

    # Data selection
    # Whiat data to use (visual, audial, all together)
    test_config.visual          = None

    # Set to True to enable binary classification mode
    test_config.binary          = False

    if test_config.binary:

        # What person to use to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = 1

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = True

        # What phonemes to use during train
        # Set to None to train on each pair combination
        # Set to pair to train on this data pair
        test_config.phoneme_classes = None

    else:

        # List of persons to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = None

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = False

        # Phoneme classes to use during train
        # Set to None to use all phonemes
        test_config.phoneme_classes = None

    # Force enable/disable dataset reloading, for example, when performing multiple restarts
    RELOAD_DATASET = True

    # Train properties
    test_config.test_size         = 0.2
    test_config.batch_size        = 8
    test_config.epochs            = 75
    test_config.use_conv_sigmoid  = False
    test_config.use_dense_sigmoid = False  # test_config.binary

    # Train autoconfig
    test_config.lr_step_size     = 40
    test_config.lr_start         = 0.01

    # Transform properties
    test_config.shift_transform_scale = 1.1
    test_config.shift_transform_roll  = 1.0
    test_config.noise_transform_scale = 0.002
    test_config.make_flip_along_time  = False

    test_config.seed       = 42

    # Model properties
    if not LOCALIZED:
        test_config.conv_layers = (
            {
                'out': 4,
                'kernel': (1, 4),
                'pool': (1, 4),
                'dropout': 0.001
            },
            {
                'out': 8,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 16,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 128,
                'dropout': 0
            },
        )
    else:
        test_config.conv_layers = (
            {
                'out': 13,
                'kernel': (1, 4),
                'pool': (1, 4),
                'dropout': 0.001
            },
            {
                'out': 26,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 52,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 256,
                'dropout': 0
            },
        )

###################################################

# Enable / disable config
MULTICLASS_ALL = False

if MULTICLASS_ALL:
    test_config = dotdict()

    # Data selection
    # Whiat data to use (visual, audial, all together)
    test_config.visual          = None

    # Set to True to enable binary classification mode
    test_config.binary          = False

    if test_config.binary:

        # What person to use to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = 1

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = True

        # What phonemes to use during train
        # Set to None to train on each pair combination
        # Set to pair to train on this data pair
        test_config.phoneme_classes = None

    else:

        # List of persons to train on
        # Set to None to train on each person
        # Set to number to train on single person
        test_config.person          = None

        # Set to True to combine data from all persons into single dataset
        test_config.all_person_data = True

        # Phoneme classes to use during train
        # Set to None to use all phonemes
        test_config.phoneme_classes = None

    # Force enable/disable dataset reloading, for example, when performing multiple restarts
    RELOAD_DATASET = False

    # Train properties
    test_config.test_size         = 0.1
    test_config.batch_size        = 16
    test_config.epochs            = 80
    test_config.use_conv_sigmoid  = False
    test_config.use_dense_sigmoid = False  # test_config.binary

    # Train autoconfig
    test_config.lr_step_size     = 40
    test_config.lr_start         = 0.01

    # Transform properties
    test_config.shift_transform_scale = 1.1
    test_config.shift_transform_roll  = 1.0
    test_config.noise_transform_scale = 0.002
    test_config.make_flip_along_time  = False

    test_config.seed       = 42

    # Model properties
    if not LOCALIZED:
        test_config.conv_layers = (
            {
                'out': 4,
                'kernel': (1, 4),
                'pool': (1, 4),
                'dropout': 0.001
            },
            {
                'out': 8,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 16,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 256,
                'dropout': 0
            },
            {
                'count': 128,
                'dropout': 0
            },
        )
    else:
        test_config.conv_layers = (
            {
                'out': 13,
                'kernel': (1, 4),
                'pool': (1, 4),
                'dropout': 0.001
            },
            {
                'out': 26,
                'kernel': (3, 5),
                'pool': (1, 2),
                'dropout': 0.001
            },
            {
                'out': 52,
                'kernel': (3, 3),
                'pool': (2, 2),
                'dropout': 0.001
            },
        )

        test_config.dense_layers = (
            {
                'count': 256,
                'dropout': 0
            },
            {
                'count': 128,
                'dropout': 0
            },
        )
