from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args=option.parse_args()
from model import Model
from dataset import Dataset
from torchinfo import summary
import umap
import numpy as np
#import time
import sys

# MODEL_LOCATION = 'saved_models/'
# MODEL_NAME = '888tiny'
# MODEL_EXTENSION = '.pkl'

MODEL_LOCATION = 'ckpt/'
MODEL_NAME = 'modelfinal'
MODEL_EXTENSION = '.pkl'

def test(dataloader, model, args, device = 'cuda', name = "training", main = False):
    model.to(device)
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = []
        labels = []
        feats = []
        #time_start = time.time()
        for _, inputs in tqdm(enumerate(dataloader)):
            labels += inputs[1].cpu().detach().tolist()
            input = inputs[0].to(device)
            scores, feat = model(input)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            pred_ = scores.cpu().detach().tolist()
            feats += feat.cpu().detach().tolist()
            pred += pred_
        #print("Time taken to process " + str(len(dataloader)) + " inputs: " + str(time.time() - time_start))
        fpr, tpr, threshold = roc_curve(labels, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(labels, pred)
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('roc_auc : ' + str(roc_auc))

        if main:
            feats = np.array(feats)
            fit = umap.UMAP()
            reduced_feats = fit.fit_transform(feats)
            labels = np.array(labels)
            plt.figure()
            plt.scatter(reduced_feats[labels == 0,0], reduced_feats[labels == 0,1], c='tab:blue', label='Normal', marker = 'o')
            plt.scatter(reduced_feats[labels == 1,0], reduced_feats[labels == 1,1], c='tab:red', label='Anomaly', marker = '*')
            plt.title('UMAP Embedding of Video Features')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.legend()
            plt.savefig(name + "_embed.png", bbox_inches='tight')
            plt.close()
        
        return roc_auc, pr_auc


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2025)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.manual_seed(2025)
        print("Using CPU - no CUDA detected")

    args = option.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    if args.model_arch == 'base':
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch in ['fast', 'tiny']:
        model = Model(
            dropout = args.dropout_rate,
            attn_dropout = args.attn_dropout_rate,
            ff_mult = 1,
            dims = (32, 32),
            depths = (1, 1),
            block_types = ('r', 'a')   # <-- must match training config!
        )
    else:
        print("Model architecture not recognized")
        sys.exit()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    summary(model, (1, 192, 16, 10, 10))
    state_dict = torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION, map_location=device)
    model.load_state_dict(state_dict, strict=False) # We may need to remove strict=False

    auc = test(test_loader, model, args, device, name = MODEL_NAME, main = True)
