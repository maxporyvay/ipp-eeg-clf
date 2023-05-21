import torch
from classifier.network_model import MNetwork


def calculate_match_on_dataset(model: MNetwork, test_dataset):
    count = 0
    with torch.no_grad():
        model.eval()
        for morlet, label in test_dataset:
            output = model(morlet[None, ...].float())
            if output.shape[-1] == 1:
                count += ((output > 0.5).int() == label).sum().item()
            else:
                predicted = torch.argmax(output, 1)
                count += (predicted == label).sum().item()

    return count / len(test_dataset)


def calculate_fail_on_dataset(model: MNetwork, test_dataset):
    count = 0
    with torch.no_grad():
        model.eval()
        for morlet, label in test_dataset:
            output = model(morlet[None, ...].float())
            if output.shape[-1] == 1:
                count += ((output > 0.5).int() != label).sum().item()
            else:
                predicted = torch.argmax(output, 1)
                count += (predicted != label).sum().item()

    return count / len(test_dataset)
