import torch
import wandb
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

config = dict(
    epochs=500,
    batch_size=128,
    val_split=8000,
    learning_rate=0.001,
    log_interval=10,
    val_interval=1
)
wandb.login()





def shuffle_images_targets(images, targets):
    perm = torch.randperm(targets.shape[0])
    images = images[perm]
    targets = targets[perm]
    return images, targets


class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(1, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 2)
        self.fc4 = torch.nn.Linear(2, 2)
        self.fc5 = torch.nn.Linear(2, 2)
        self.fc6 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, images, targets, transform=None, target_transform=None, shuffle=False):
        if shuffle:
            images, targets = shuffle_images_targets(images, targets)
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

def train(model, train_loader, val_loader, optimizer, criterion, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, targets) in enumerate(train_loader):

            loss = train_batch(model, images, targets, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % config.log_interval) == 0:
                train_log(loss, example_ct, epoch)
        if epoch % config.val_interval == 0:
            val_loss = validation_test(model, val_loader, criterion)
            val_log(val_loss, example_ct, epoch)

    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

def validation_test(model, loader, criterion):
    losses = []
    with torch.no_grad():
        for _, (images, targets) in enumerate(loader):
            predicted_target = model(images)
            loss = criterion(predicted_target, targets)
            losses.append(loss)
    mean_loss = np.stack(losses).mean().item()
    return mean_loss

def train_batch(model, image, target, optimizer, criterion):
    predicted_target = model(image)
    loss = criterion(predicted_target, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    # print(f"Train loss after {str(example_ct).zfill(5)} examples: {loss:.6f}")


def val_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    # print(f"Validation loss after {str(example_ct).zfill(5)} examples: {loss:.6f}")

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config):
    model = Network()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    images = torch.linspace(0, 1, 10000)[..., None]
    targets = 2 * images ** 2 - images  - 1
    images, targets = shuffle_images_targets(images, targets)
    train_dataset = CustomImageDataset(images[:config.val_split], targets[:config.val_split])
    val_dataset = CustomImageDataset(images[config.val_split:], targets[config.val_split:])
    train_loader = make_loader(train_dataset, config.batch_size)
    val_loader = make_loader(val_dataset, config.batch_size)
    return model, train_loader, val_loader, optimizer, criterion


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, val_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, val_loader, criterion, optimizer, config)


def main():
    model_pipeline(config)

if __name__ == '__main__':
    main()

