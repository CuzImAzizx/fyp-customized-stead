import torch
import torch.nn.functional as F
import option
args=option.parse_args()
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

torch.autograd.set_detect_anomaly(True)

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)
        return d

    def forward(self, feats, margin = 100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(torch.zeros(bs // 2).cuda(), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)

class Loss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.1, use_supcon=False):
        super(Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()  # Make sure this supports hard mining
        self.alpha = alpha
        self.beta = beta
        self.use_supcon = use_supcon

    def supervised_contrastive_loss(self, features, labels, temperature=0.07):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)

        # For stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(self, scores, feats, targets):
        """
        scores: [B, 1] raw logits
        feats: [B, C] feature embeddings
        targets: [B] binary labels (0 or 1)
        """
        loss_ce = self.criterion(scores, targets.float())

        loss_triplet = self.triplet(feats, targets)  # You may need anchors/positives/negatives
        loss = loss_ce + self.alpha * loss_triplet

        if self.use_supcon:
            loss_supcon = self.supervised_contrastive_loss(feats, targets)
            loss += self.beta * loss_supcon
            return loss, loss_ce, loss_triplet, loss_supcon

        return loss, loss_ce, loss_triplet

def train(loader, model, optimizer, scheduler, device, epoch):

    with torch.set_grad_enabled(True):
        model.train()
        pred = []
        label = []
        for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader)):
            input = torch.cat((ninput, ainput), 0).to(device)
            
            scores, feats, = model(input) 
            pred += scores.cpu().detach().tolist()
            labels = torch.cat((nlabel, alabel), 0).to(device)
            label += labels.cpu().detach().tolist()

            loss_criterion = Loss()
            loss_ce, loss_con = loss_criterion(scores.squeeze(), feats, labels)
            loss = loss_ce + loss_con

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step_update(epoch * len(loader) + step)
        fpr, tpr, _ = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, pred)
        pr_auc = auc(recall, precision)
        print('train_pr_auc : ' + str(pr_auc))
        print('train_roc_auc : ' + str(roc_auc))
        return  loss.item()
