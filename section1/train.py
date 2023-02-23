import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


class CustomScaler:
    def __init__(self, scale_factor: float = 2 ** 10, scaler_type: str = "dynamic", zero_cutoff_share: int = 0.0001):
        self.scale_factor = scale_factor
        self.counter = 0
        self.scaler_type = scaler_type
        self.zero_cutoff_share = zero_cutoff_share

    def scale(self, loss):
        return loss * self.scale_factor

    def count_inf_grads(self, optimizer):
        inf_grads = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    inf_grads += (torch.logical_not(torch.isfinite(param.grad))).sum().item()

        return inf_grads

    def count_zero_grads(self, optimizer):
        zero_grads = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    zero_grads += int(torch.count_nonzero(param.grad))
        return zero_grads

    def count_total_grads(self, optimizer):
        total_grads = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    total_grads += param.grad.isnumel()
        return total_grads

    def unscale_grads(self, optimizer):
        for g in optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    p.grad /= self.scale_factor

    def step(self, optimizer):
        if self.scaler_type == "dynamic":
            total_grad = self.count_total_grads(optimizer)
            inf_grads_share = self.count_inf_grads(optimizer) / total_grad
            zero_grads_share = self.count_zero_grads(optimizer) / total_grad
            if inf_grads_share > 0:
                self.scale_factor /= 2
                return
            elif zero_grads_share > self.zero_cutoff_share:
                self.scale_factor *= 2

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
    acc_list = []
    loss_list = []
    scale_factors_list = []

    model.train()

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

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")

    acc_list.append(accuracy.item())
    loss_list.append(loss.item())

    return acc_list, loss_list, scale_factors_list


def train(scaler):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = get_train_data()
    num_epochs = 5

    acc_list = []
    loss_list = []
    scale_factors_list = []

    for epoch in range(0, num_epochs):
        acc, loss, scale_factors = train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)
        acc_list += acc
        loss_list += loss
        scale_factors_list += scale_factors

    return acc_list, loss_list, scale_factors_list