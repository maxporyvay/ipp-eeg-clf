import torch
import json

from classifier.config import TARGET_CHANNELS, SECTOR_LENGTH_STEPS, MORLET_FREQ_STEPS


class MNetwork(torch.nn.Module):
    """
    Base blass for morlet classification network.

    conv_layers - list of dicts defining each convolutional layer
    """

    def __init__(self, conv_layers, dense_layers, num_classes, use_conv_sigmoid=False, use_dense_sigmoid=False, print_log=True):
        super().__init__()

        # Conv layers
        self.conv = []
        # MaxPool layers
        self.pool = []
        # Dropout for layers
        self.conv_dropout = []
        # Fully connected layers
        self.fc = []
        # Dropout for layers
        self.fc_dropout = []

        # Use sigmoid or relu
        self.use_conv_sigmoid = use_conv_sigmoid
        self.use_dense_sigmoid = use_dense_sigmoid

        # Save useful parameters
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.num_classes = num_classes

        # Add all conv layers
        # Input conv layer 'in' parameter is ignored and equals to TARGET_CHANNELS x MORLET_FREQ_STEPS x SECTOR_LENGTH_STEPS
        # Kernel is tuple
        # pool is tuple
        last_conv_out = TARGET_CHANNELS
        result_dimension = [TARGET_CHANNELS, MORLET_FREQ_STEPS, SECTOR_LENGTH_STEPS]
        for i, layer in enumerate(conv_layers):
            self.conv.append(torch.nn.Conv2d(last_conv_out, layer['out'], layer['kernel']))
            self.__setattr__(f'conv{i}', self.conv[-1])
            last_conv_out = layer['out']

            self.pool.append(torch.nn.MaxPool2d(layer['pool']))
            self.__setattr__(f'pool{i}', self.pool[-1])

            self.conv_dropout.append(torch.nn.Dropout(layer['dropout']))
            self.__setattr__(f'conv_dropout{i}', self.conv_dropout[-1])

            # Recalculate dimensions
            result_dimension[0] = layer['out']
            result_dimension[1] = result_dimension[1] - layer['kernel'][0] + 1
            result_dimension[2] = result_dimension[2] - layer['kernel'][1] + 1
            result_dimension[1] = result_dimension[1] // layer['pool'][0]
            result_dimension[2] = result_dimension[2] // layer['pool'][1]

        if print_log:
            print('conv-dense dimension:', result_dimension)

        # Add fully connected
        # dense_layers contain only layers sizes between input and output of dense net
        # Input of the dense has dimensions of the last conv layer
        # Outputs of the dense has dimensions of the num_classes
        last_fc_out = result_dimension[0] * result_dimension[1] * result_dimension[2]
        for i, layer in enumerate(dense_layers):
            self.fc.append(torch.nn.Linear(last_fc_out, layer['count']))
            self.__setattr__(f'fc{i}', self.fc[-1])
            last_fc_out = layer['count']

            self.fc_dropout.append(torch.nn.Dropout(layer['dropout']))
            self.__setattr__(f'fc_dropout{i}', self.fc_dropout[-1])

        # Append last fc layer
        self.fc.append(torch.nn.Linear(last_fc_out, num_classes))

    def forward(self, x):
        # Apply conv
        for conv, pool, drop in zip(self.conv, self.pool, self.conv_dropout):
            if self.use_conv_sigmoid:
                x = drop(pool(torch.sigmoid(conv(x))))
            else:
                x = drop(pool(torch.relu(conv(x))))

        # Flatten
        x = torch.flatten(x, 1)

        # Dense
        for fc, drop in zip(self.fc[:-1], self.fc_dropout):
            if self.use_dense_sigmoid:
                x = drop(torch.sigmoid(fc(x)))
            else:
                x = drop(torch.relu(fc(x)))

        x = self.fc[-1](x)

        return x

    def save_model(self, filename):
        config = {
            'conv_layers': self.conv_layers,
            'dense_layers': self.dense_layers,
            'num_classes': self.num_classes,
            'use_conv_sigmoid': self.use_conv_sigmoid,
            'use_dense_sigmoid': self.use_dense_sigmoid
        }

        # Save config
        with open(f'{filename}-config.json', 'w') as f:
            json.dump(config, f)

        # Save all parameters
        for i, conv in enumerate(self.conv):
            state = conv.state_dict()

            for key in state.keys():
                torch.save(state[key], f'{filename}-conv{i}-{key}.tensor')

        # Save all parameters
        for i, fc in enumerate(self.fc):
            state = fc.state_dict()

            for key in state.keys():
                torch.save(state[key], f'{filename}-fc{i}-{key}.tensor')


def load_model(filename):
    # Save config
    with open(f'{filename}-config.json', 'r') as f:
        config = json.load(f)

    # Create model
    model = MNetwork(config['conv_layers'], config['dense_layers'], config['num_classes'], config['use_conv_sigmoid'], config['use_dense_sigmoid'])

    # Save all parameters
    for i, conv in enumerate(model.conv):
        state = conv.state_dict()

        for key in state.keys():
            state[key] = torch.load(f'{filename}-conv{i}-{key}.tensor')

        conv.load_state_dict(state)

    # Save all parameters
    for i, fc in enumerate(model.fc):
        state = fc.state_dict()

        for key in state.keys():
            state[key] = torch.load(f'{filename}-fc{i}-{key}.tensor')

        fc.load_state_dict(state)

    return model
