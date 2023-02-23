import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


class СustomScaler:
    def __init__(self, scale_factor: float = 2 ** 10, scaler_type: str = "dynamic", double_interval: int = 100):
        self.scale_factor = scale_factor
        self.counter = 0
        self.scaler_type = scaler_type
        self.double_interval = double_interval

    def scale(self, loss):
        return loss * self.scale_factor

    @staticmethod
    def count_inf_grads(optimizer):
        inf_grads = 0
        for g in optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    inf_grads += (torch.logical_not(torch.isfinite(p.grad))).sum().item()

        return inf_grads

    def unscale_grads(self, optimizer):
        for g in optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    p.grad /= self.scale_factor

    def step(self, optimizer):

        if self.scaler_type == "dynamic":
            inf_grads = self.count_inf_grads(optimizer)
            if inf_grads > 0:
                self.scale_factor /= 2
                self.counter = 0
            else:
                self.counter += 1
                if self.counter > self.double_interval:
                    self.scale_factor *= 2
                    self.counter = 0

                self.unscale_grads(optimizer)
                optimizer.step()
        else:
            self.unscale_grads(optimizer)
            optimizer.step()


def train_epoch(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler
):
    model.train()
    acc_list = []
    loss_list = []
    scale_factors_list = []


    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scale_factors_list.append(scaler.scale_factor)

        accuracy = ((outputs > 0.5) == labels).float().mean()

        acc_list.append(accuracy)
        loss_list.append(loss.item())

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")

        return acc_list, loss_list, scale_factors_list

def train(scaler):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = get_train_data()
    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)
