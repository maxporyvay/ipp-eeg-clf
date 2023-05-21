import torch
import numpy as np
import torchvision
import json

from classifier.dataset_training_utils import load_dataset, select_phonemes, normalize_morlets, train_test_split
from classifier.transform import NoiseTransform, ResizeShiftTransform, FlipAlongTime, ToTensor
from classifier.dataset import MorletDataset
from classifier.network_model import MNetwork
from classifier.model_evaluation import calculate_match_on_dataset
from classifier.model_training_fun import train_model

from classifier.config import SAVE_CONFIG, INPUT_EDF_LIST, PHONEME_COUNT, SAVE_CHECKPOINTS


def get_person_list(test_config):
    # What data to use during train
    if test_config.all_person_data:
        # List of person IDS to fetch data from
        persons_list = [None]
        print('Combine all person data')

    else:
        if test_config.person is None:
            persons_list = list(range(len(INPUT_EDF_LIST)))
            print('Train on each person separately')
        else:
            persons_list = [test_config.person]
            print('Train on single person:', test_config.person)

    return persons_list


def get_phonemes_list(test_config):
    # What phoneme combinations to use
    if test_config.binary:
        if test_config.phoneme_classes is not None:
            # Binary classifier for given class numbers
            phoneme_list = [test_config.phoneme_classes]
            print('Train on single phoneme pair:', test_config.phoneme_classes)
        else:
            # Binary classifier for every pair of classes
            phoneme_list = []
            for i in range(PHONEME_COUNT):
                for j in range(PHONEME_COUNT):
                    if i > j:
                        phoneme_list.append((i, j))
            print('Train on each phoneme pair')
    else:
        # Select data for given list of classes
        phoneme_list = [None]

        if test_config.phoneme_classes is None:
            print('Train on all phoneme classes')
        else:
            print('Train on selected phoneme classes:', test_config.phoneme_classes)

    return phoneme_list


def load_data(test_config, person, phonemes_pair, base_labels, base_morlets, seed):
    print()
    print()
    if test_config.binary:
        print('person =', person, 'phoneme1 =', phonemes_pair[0], 'phoneme2 =', phonemes_pair[1])
    else:
        print('person =', person, 'classes =', test_config.phoneme_classes)

    # Phoneme classes to select from data
    if test_config.binary:
        phonemes_classes = phonemes_pair
        num_classes = 1
    else:
        if test_config.phoneme_classes is None:
            phonemes_classes = None  # list(range(PHONEME_COUNT))
            num_classes = PHONEME_COUNT
        else:
            phonemes_classes = test_config.phoneme_classes
            num_classes = len(test_config.phoneme_classes)

    # Load dataset & select phonemes
    labels, morlets = select_phonemes(base_labels, base_morlets, phonemes_classes)
    morlets = normalize_morlets(morlets)

    print('labels stats:')
    stats = np.unique(labels, return_counts=True)
    print('\n'.join([f'{v[0]}: {v[1]}' for v in zip(stats[0], stats[1])]))

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split
    train_labels, train_morlets, test_labels, test_morlets = train_test_split(labels, morlets, test_config.test_size)

    print('train labels stats:')
    stats = np.unique(train_labels, return_counts=True)
    print('\n'.join([f'{v[0]}: {v[1]}' for v in zip(stats[0], stats[1])]))

    print('test labels stats:')
    stats = np.unique(test_labels, return_counts=True)
    print('\n'.join([f'{v[0]}: {v[1]}' for v in zip(stats[0], stats[1])]))

    print('Train count:', len(train_labels))
    print('Test count: ', len(test_labels))

    # Transform
    train_transforms = [
        ResizeShiftTransform(test_config.shift_transform_scale, test_config.shift_transform_roll),
        NoiseTransform(test_config.noise_transform_scale)
    ]
    if test_config.make_flip_along_time:
        train_transforms.append(FlipAlongTime())
    train_transforms.append(ToTensor())

    train_transform = torchvision.transforms.Compose(train_transforms)

    test_transform = torchvision.transforms.Compose([
        ToTensor()
    ])

    # Dataset
    train, test = MorletDataset(train_labels, train_morlets, train_transform), MorletDataset(test_labels, test_morlets, test_transform)

    # DataLoader for train
    train_loader = torch.utils.data.DataLoader(train, batch_size=test_config.batch_size, shuffle=True)

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    return num_classes, train_loader, train, test


def do_training(test_config, num_classes, train_loader, train, test):
    print('num_classes', num_classes)

    # Create model
    model = MNetwork(
        test_config.conv_layers,
        test_config.dense_layers,
        num_classes,
        use_conv_sigmoid=test_config.use_conv_sigmoid,
        use_dense_sigmoid=test_config.use_dense_sigmoid
    )
    print('Network:', 'conv_layers', test_config.conv_layers, 'dense_layers', test_config.dense_layers, 'num_classes', num_classes, 'use_conv_sigmoid', test_config.use_conv_sigmoid, 'use_dense_sigmoid', test_config.use_dense_sigmoid)

    print(f'Train size:    {len(train)}')
    print(f'Test size:     {len(test)}')

    # Training requirements
    if num_classes > 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=test_config.lr_start, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_config.lr_step_size, gamma=0.1)

    # Train
    train_model(model, train_loader, optimizer, scheduler, criterion, test_config.epochs, print_log=True, print_iters=100, print_last_iter=False)

    # Evaluate
    result_match = calculate_match_on_dataset(model, test) * 100
    print(f'After train match: {round(result_match, 2)}%')

    return model, result_match


def save_data(test_config, model, person, phonemes_pair, seed, result_match):
    if test_config.binary:
        checkpoint_name = f'checkpoint-{test_config.checkpoint_prefix}-{test_config.visual}-{person}-binary-ph_{phonemes_pair[0]}-ph_{phonemes_pair[1]}-ep_{test_config.epochs}-seed_{seed}'
    else:
        checkpoint_name = f'checkpoint-{test_config.checkpoint_prefix}-{test_config.visual}-{person}-multiclass-phs_{"_".join(test_config.phoneme_classes or [ "all" ])}-ep_{test_config.epochs}-seed_{seed}'
    model.save_model(f'checkpoints/{checkpoint_name}')

    checkpoint_info = {}
    try:
        with open(test_config.test_json, 'r') as f:
            checkpoint_info = json.load(f)
    except Exception:
        pass

    checkpoint_info[checkpoint_name] = {
        'result_match': result_match
    }
    try:
        with open(test_config.test_json, 'w') as f:
            json.dump(checkpoint_info, f)
    except Exception:
        print('Saving error')
        print(checkpoint_info)


def run_test(test_config):
    if SAVE_CONFIG:
        with open(test_config.test_config_json, 'w') as f:
            json.dump(test_config, f)

    print('binary:', test_config.binary)

    person_list = get_person_list(test_config)
    phonemes_list = get_phonemes_list(test_config)

    for person in person_list:
        print('person:', person)
        print('test_config.visual:', test_config.visual)
        base_labels, base_morlets = load_dataset(test_config.visual, person)
        print('labels:', set(base_labels))

        for phonemes_pair in phonemes_list:
            seed = test_config.seed

            num_classes, train_loader, train, test = load_data(test_config, person, phonemes_pair, base_labels, base_morlets, seed)

            model, result_match = do_training(test_config, num_classes, train_loader, train, test)

            if SAVE_CHECKPOINTS:
                save_data(test_config, model, person, phonemes_pair, seed, result_match)
