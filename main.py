from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option
args=option.parse_args()
from model import Model
from dataset import Dataset
from train import train
from test import test
import datetime
import os
import random
import sys


def save_config(save_path):
    path = save_path+'/'
    os.makedirs(path,exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    f = open(path + f"config_{timestamp}.txt", "w")
    for key in vars(args).keys():
        f.write('{}: {}'.format(key,vars(args)[key]))
        f.write('\n')

savepath = './ckpt/{}_{}_{}'.format(args.lr, args.batch_size, args.comment)
save_config(savepath)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
if __name__ == '__main__':
    args=option.parse_args()
    random.seed(2025)
    np.random.seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)
    else:
        torch.manual_seed(2025)  # fall back safely for CPU

    # DO NOT SHUFFLE, shuffling is handled by the Dataset class and not the DataLoader
    train_loader = DataLoader(Dataset(args, test_mode=False),
                               batch_size=args.batch_size // 2)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size)

    if args.model_arch == 'base':
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch in ['fast', 'tiny']:
        model = Model(
            dropout = args.dropout_rate,
            attn_dropout = args.attn_dropout_rate,
            ff_mult = 1,
            
            # Update these
            dims = (32, 64),
            depths = (1, 1), 
            block_types = ('t', 'a')   # <— EfficientNet + Attention
        )
    else:
        print("Model architecture not recognized")
        sys.exit()

    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(
            torch.load(args.pretrained_ckpt, map_location=device)
        )
        print("pretrained loaded")

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial= args.max_epoch * num_steps,
            cycle_mul=1.,
            lr_min=args.lr * 0.2,
            warmup_lr_init=args.lr * 0.01,
            warmup_t=args.warmup * num_steps,
            cycle_limit=20,
            t_in_epochs=False,
            warmup_prefix=True,
            cycle_decay = 0.95,
        )

    best_pr_auc = 0
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_roc_list = []
    val_roc_list = []

    criterion = torch.nn.BCEWithLogitsLoss()

    for step in tqdm(range(0, args.max_epoch), total=args.max_epoch, dynamic_ncols=True):

        # ---- TRAIN ----
        train_loss, train_roc_auc = train(train_loader, model, optimizer, scheduler, device, step)
        train_losses.append(train_loss)
        train_roc_list.append(train_roc_auc)

        # ---- VALIDATION ----
        model.eval()
        all_labels, all_scores = [], []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                scores, feats = model(inputs)
                val_loss += criterion(scores.squeeze(), labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.sigmoid(scores).cpu().numpy())

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        # Compute validation ROC-AUC
        from sklearn.metrics import roc_auc_score
        val_roc_auc = roc_auc_score(all_labels, all_scores)
        val_roc_list.append(val_roc_auc)

        # ---- LOGGING ----
        print(f"Epoch [{step+1}/{args.max_epoch}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train ROC-AUC: {train_roc_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")

        # ---- EARLY STOPPING ----
        if val_roc_auc > best_pr_auc:
            best_pr_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), savepath + '/BEST_MODEL.pkl')
            print(f"New best model saved | Val ROC-AUC = {val_roc_auc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("========= Early stopping activated. Training stopped. =========")
            break

        scheduler.step(step + 1)

    # ---- PLOTTING ----
    import matplotlib.pyplot as plt

    # LOSS CURVE
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath + "/loss_curve.png", bbox_inches='tight')

    # ROC-AUC CURVE
    plt.figure()
    plt.plot(train_roc_list, label="Train ROC-AUC")
    plt.plot(val_roc_list, label="Validation ROC-AUC")
    plt.title("ROC-AUC Curve")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath + "/roc_auc_curve.png", bbox_inches='tight')

    print("========= Training finished =========")
    print("Best model saved at:", savepath + '/BEST_MODEL.pkl')
    print("Charts saved in:", savepath)

