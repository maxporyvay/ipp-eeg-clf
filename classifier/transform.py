import cv2
import numpy as np
import random
import torch

from classifier.config import SECTOR_LENGTH_STEPS


class NoiseTransform(object):
    """
    Add noise to the sample
    """

    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def __call__(self, sample):
        morlet = sample
        return morlet + np.random.normal(0, self.noise_scale, morlet.shape)


class ResizeShiftTransform(object):
    """
    Perform rescale of the meorlet and shifting it in a random position
    """

    def __init__(self, max_scale, max_roll):
        self.max_scale = max_scale
        self.max_roll = max_roll
        self.max_pad = int(SECTOR_LENGTH_STEPS * (max_scale - 1))

    def __call__(self, sample):
        morlet = sample.transpose(1, 2, 0)
        rand_pad = random.randint(0, self.max_pad)
        resized = cv2.resize(morlet, dsize=(morlet.shape[1] + rand_pad, morlet.shape[0]), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

        roll = np.roll(resized, -random.randint(0, int(rand_pad * self.max_roll)), 2)

        crop = roll[:, :, 0:SECTOR_LENGTH_STEPS]

        return crop


class FlipAlongTime(object):
    """
    Flip data along time axis
    """

    def __call__(self, sample):
        return np.flip(sample, axis=2) if random.randint(0, 1) else sample


class ToTensor(object):
    """
    Convert ndarray to tensor
    """

    def __call__(self, sample):
        morlet = sample

        return torch.from_numpy(morlet.copy())
