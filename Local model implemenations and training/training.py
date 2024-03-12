import torch, colossalai
from pathlib import Path
from colossalai.booster import Booster
from colossalai.zero import ColoInitContext
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from torchMixnet import build_Mixnet
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import tqdm
import pandas as pd
import scipy, librosa
import numpy as np
import pywt
import torch.utils.data as data


# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3



NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


def spectrogram_from_eeg(parquet_path, eeg_offset, eeg_id):

    # LOAD MIDDLE 8.533 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = eeg_offset+5000
    eeg = eeg.iloc[middle-853:middle+853]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,512,4),dtype='float32')
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # UNDERSAMPLE SIGNAL 200Hz to 60Hz (minimum sr for not loosing information on the 0-30Hz bandwith (shanon-nyquist))
            x  = scipy.signal.resample(x, 512)

            # RAW SPECTROGRAM
            fs = 60
            frequencies = np.linspace(0.2, 30, 128) / fs # normalize
            scale = pywt.frequency2scale('cmor1-0.5', frequencies)
            scalo , freqs = pywt.cwt(x, scale, 'cmor1-0.5', sampling_period = 1/fs)

            # LOG TRANSFORM
            width = (scalo.shape[1]//32)*32
            scalo_db = librosa.power_to_db(np.abs(scalo), ref=0).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            # scalo_db = (scalo_db+40)/40

            img[:,:,k] += scalo_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

    return img

class IterDataset(data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()



def build_dataloader(batch_size: int, plugin, train_MD_df: pd.DataFrame, size: int):

    def train_dataset_generator():
        for index, row in (train_MD_df.iloc[:size*0.8]).iterrows():
            eeg_offset = int( row['eeg_label_offset_seconds'] )
            parquet_path = row['path_eeg']
            eeg_id = row['eeg_id']
            img = spectrogram_from_eeg(parquet_path, eeg_offset, eeg_id, display=False)
            yield img, row['y']

    def test_dataset_generator():
        for index, row in (train_MD_df.iloc[size*0.8:size]).iterrows():
            eeg_offset = int( row['eeg_label_offset_seconds'] )
            parquet_path = row['path_eeg']
            eeg_id = row['eeg_id']
            img = spectrogram_from_eeg(parquet_path, eeg_offset, eeg_id, display=False)
            yield img, row['y']


    train_dataset = IterDataset(train_dataset_generator)
    test_dataset = IterDataset(test_dataset_generator)

    train_dataset = transforms.ToTensor()(train_dataset)
    test_dataset = transforms.ToTensor()(test_dataset)

    # Data loader
    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader


@torch.no_grad()
def evaluate(model: nn.Module, test_dataloader: DataLoader) -> float:
    model.eval()
    correct = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    total = torch.zeros(1, dtype=torch.int64, device=get_accelerator().get_current_device())
    for images, labels in test_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct.item() / total.item()
    print(f"Accuracy of the model on the test images: {accuracy * 100:.2f} %")
    return accuracy

def train_epoch(epoch: int,
                model: nn.Module,
                optimizer: Optimizer,
                criterion: nn.Module,
                train_dataloader: DataLoader,
                booster: Booster):
    model.train()
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]") as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({"loss": loss.item()})

def main(plugin_, resume = -1, interval = 5, checkpoint = "./checkpoint", target_acc = None):

    # ==============================
    # Prepare Checkpoint Directory
    # ==============================
    if interval > 0:
        Path(checkpoint).mkdir(parents=True, exist_ok=True)

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(config={})

    # update the learning rate
    global LEARNING_RATE

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================


    booster = Booster(plugin=plugin_)

    # ==============================
    # Prepare Dataloader
    # ==============================
    train_dataloader, test_dataloader = build_dataloader(100, plugin)

    # ====================================
    # Prepare model, optimizer, criterion
    # ====================================
    # resent50
    model = build_Mixnet()

    # Loss and optimizer
    criterion = nn.KLDivLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)

    # lr scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    )

    # ==============================
    # Resume from checkpoint
    # ==============================
    if resume >= 0:
        booster.load_model(model, f"{checkpoint}/model_{resume}.pth")
        booster.load_optimizer(optimizer, f"{checkpoint}/optimizer_{resume}.pth")
        booster.load_lr_scheduler(lr_scheduler, f"{checkpoint}/lr_scheduler_{resume}.pth")

    # ==============================
    # Train model
    # ==============================
    start_epoch = resume if resume >= 0 else 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, criterion, train_dataloader, booster)
        lr_scheduler.step()

        # save checkpoint
        if interval > 0 and (epoch + 1) % interval == 0:
            booster.save_model(model, f"{checkpoint}/model_{epoch + 1}.pth")
            booster.save_optimizer(optimizer, f"{checkpoint}/optimizer_{epoch + 1}.pth")
            booster.save_lr_scheduler(lr_scheduler, f"{checkpoint}/lr_scheduler_{epoch + 1}.pth")

    accuracy = evaluate(model, test_dataloader)
    if target_acc is not None:
        assert accuracy >= target_acc, f"Accuracy {accuracy} is lower than target accuracy {target_acc}"


if __name__ == "__main__":
    plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, max_norm=1.0, initial_scale=2**5)
    main(plugin, resume = -1, interval = 5, checkpoint = "./checkpoint", target_acc = None)
