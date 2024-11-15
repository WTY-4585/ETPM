from ultis.earlyrnn_500 import EarlyRNN
import torch
from tqdm import tqdm
from ultis.earlystop_loss import EarlyRewardLoss
import numpy as np
from ultis.visdom import VisdomLogger
from ultis.readdatalabel import *
import sklearn.metrics
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Run Early Classification training...')
    parser.add_argument('--alpha', type=float, default=0.6, help="trade-off parameter of earliness and accuracy (eq 6): "
                                                                 "1=full weight on accuracy; 0=full weight on earliness")
    parser.add_argument('--epsilon', type=float, default=10, help="additive smoothing parameter that helps the "
                                                                  "model recover from too early classificaitons (eq 7)")
    parser.add_argument('--learning-rate', type=float, default=1e-2, help="Optimizer learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight_decay")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--epochs', type=int, default=200, help="number of training epochs")  # 100
    parser.add_argument('--sequencelength', type=int, default=500, help="sequencelength of the time series. If samples are shorter, "
                                                                "they are zero-padded until this length; "
                                                                "if samples are longer, they will be undersampled")
    parser.add_argument('--batchsize', type=int, default=256, help="number of samples per batch")
    parser.add_argument('--snapshot', type=str, default="./snapshots/model.pth", help="pytorch state dict snapshot file")
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.patience < 0:
        args.patience = None
    return args


def expand_dimensions(input_tensor, target_dim=500):
    original_dim = input_tensor.size()
    expanded_tensor = input_tensor.unsqueeze(-1).expand(*original_dim, target_dim)
    return expanded_tensor.float()


def main(args):
    traindataloader, testdataloader = create_dataloaders(data_dir='./data/30class/', batch_size=256)

    nclasses = 30
    input_dim = 32 # 32
    hidden_dim = 64
    num_layers = 2
    dropout = 0.6

    model = EarlyRNN(input_dim=input_dim, hidden_dims=hidden_dim, nclasses=nclasses, num_rnn_layers=num_layers, dropout=dropout).to(args.device)

    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": args.learning_rate}, {'params': decay}],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)

    if args.resume and os.path.exists(args.snapshot):
        print('load param.')
        model.load_state_dict(torch.load(args.snapshot, map_location=args.device))
        optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                          os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                          )
    else:
        print('train with no param')
        train_stats = []
        start_epoch = 1
    visdom_logger = VisdomLogger(log_to_filename='./snapshots/1.log')

    not_improved = 0
    with tqdm(range(start_epoch, args.epochs + 1)) as pbar:
        for epoch in pbar:
            trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=args.device)
            testloss, stats = test_epoch(model, testdataloader, criterion, args.device)

            precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0], average="macro",
                zero_division=0)
            accuracy = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0])
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0])
            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))
            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_pred=stats["predictions_at_t_stop"][:, 0],
                                                                         y_true=stats["targets"][:, 0])
            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward
                )
            )

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                     f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                     f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}. {savemsg}")

            if args.patience is not None:
                if not_improved > args.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {args.patience} epochs.")
                    break


def train_epoch(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch
        y_true = expand_dimensions(y_true)
        X, y_true = X.to(device), y_true.to(device)
        log_class_probabilities, probability_stopping = model(X)
        loss = criterion(log_class_probabilities, probability_stopping, y_true)

        if not loss.isnan().any():
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
    return np.stack(losses).mean()


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    stats = []
    losses = []
    for batch in dataloader:
        X, y_true = batch
        y_true = expand_dimensions(y_true)
        X, y_true = X.to(device), y_true.to(device)
        log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
        loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)

        stats.append(stat)

        losses.append(loss.cpu().detach().numpy())

    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}

    return np.stack(losses).mean(), stats


if __name__ == '__main__':
    args = parse_args()
    main(args)
