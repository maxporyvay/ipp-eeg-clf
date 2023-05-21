from classifier.network_model import MNetwork
import torch


def train_model(model: MNetwork, train_loader, optimizer, scheduler, criterion, train_epochs, print_log=False, print_iters=100, print_first_iter=True, print_last_iter=True):
    """
    Train model for given amount of epochs
    """

    model.train()
    for epoch in range(train_epochs):
        for iter, (morlets, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(morlets.float())

            if outputs.shape[-1] == 1:
                outputs = torch.sigmoid(outputs.flatten())
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            if print_log:
                if (iter % print_iters == 0):
                    if iter == 0 and print_first_iter:
                        print(f'epoch = {epoch}, iter = {iter}, loss = {loss.item()}')
                    elif iter > 0:
                        print(f'epoch = {epoch}, iter = {iter}, loss = {loss.item()}')
                elif print_last_iter and iter == len(train_loader) - 1:
                    print(f'epoch = {epoch}, iter = {iter}, loss = {loss.item()}')

        if scheduler is not None:
            scheduler.step()
