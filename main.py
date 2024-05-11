import numpy as np
import argparse
from tqdm import tqdm
import pickle
import os

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import *
from models import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50',
                    help='cnn/resnet34/.../resnet101')
# 完整训练次数
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
# 学习率
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
# batch_size 
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size')
# 权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
args = parser.parse_args()

train_loader, val_loader, test_loader = load_data(args.batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    train_correct = []
    train_losses = []
    train_accs = []

    val_correct = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.epochs):
        train_corr = 0
        val_corr = 0
        
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for train_b, (x_train, y_train) in loop:
            train_b += 1
            x_train = torch.as_tensor(x_train, device=device)
            y_train = torch.as_tensor(y_train, device=device)

            train_pred = model(x_train)
            train_loss = criterion(train_pred, y_train)

            train_pred_vec = torch.max(train_pred.data, 1)[1]
            train_corr += (train_pred_vec == y_train).sum()

            # Update parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # if train_b % 1000 == 0:
            #     print(f"epoch: {epoch+1:2} | batch: {train_b:4} | instances: [{train_b*args.batch_size:6} / {len(train_loader) * args.batch_size}] | loss: {train_loss.item()}")
            #     print(f"✅{train_corr.item()} of {train_b*args.batch_size:2} | accuracy: {round(((train_corr.item() / (train_b*args.batch_size)) * 100), 3)}%")

            train_correct.append(train_corr.item())
            train_losses.append(train_loss.item())

            loop.set_description(f'Train epoch [{epoch+1}/{args.epochs}]')
            loop.set_postfix(batch = train_b)
            loop.set_postfix(loss = train_loss.item())

        with torch.no_grad():
            model.eval()
            for val_b, (x_val, y_val) in enumerate(val_loader):
                val_b += 1

                x_val = torch.as_tensor(x_val, device=device)  # Convert to PyTorch tensor
                y_val = torch.as_tensor(y_val, device=device)  # Convert to PyTorch tensor

                val_pred = model(x_val)
                val_pred_vec = torch.max(val_pred.data, 1)[1]
                val_corr += (val_pred_vec == y_val).sum()

                val_loss = criterion(val_pred, y_val)
                val_correct.append(val_corr.item())
                val_losses.append(val_loss.item())

                val_acc = val_corr.item() / (len(val_loader) * args.batch_size)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"./pth/{model._get_name()}.pth")
                print(f"\tNew best model saved! | accuracy: {best_acc*100}%")

        train_epoch_acc = train_corr.item() / (args.batch_size * len(train_loader))
        val_epoch_acc = val_corr.item() / (args.batch_size * len(val_loader))

        writer.add_scalar('Train/Acc', train_epoch_acc, epoch+1)
        writer.add_scalar('Val/Acc', val_epoch_acc, epoch+1)
        writer.add_scalar('Train/Loss', np.mean(train_losses), epoch+1)
        writer.add_scalar('Val/Loss', np.mean(val_losses), epoch+1)

        train_accs.append(train_epoch_acc)
        val_accs.append(val_epoch_acc)
    
    print('Finish train.')
    # plot_loss(model_name, train_losses)
    # plot_accuracy(model_name, val_accuracies)

    torch.save({'model': model.state_dict()}, 'pth/'+model_name+'.pth')
    return train_losses, val_accs

def test(model_name, using_exist=False):
    model = get_model(model_name)
    model.to(device)
    if using_exist:
        print("loading existed params")
        state_dict = torch.load('pth/'+model_name+'.pth')
        model.load_state_dict(state_dict['model'])
    else:
        train_losses, val_accuracies = train(model)

    all_labels = []
    all_predictions = []
    all_predictions_pro = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_predictions_pro.extend(predicted_probabilities)
    
    test_accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy: {test_accuracy:.3f}')

    report = classification_report(all_labels, all_predictions)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(model_name, cm)

    plot_auroc(model_name, all_labels, all_predictions_pro)

    return train_losses, val_accuracies

model_configs = [
    'Adamresnet',
    # 'resnet18_small',
    # 'resnet18',
    # 'resnet34',
    # 'resnet50', 
    # 'resnet101',
    # 'resnet152',
    # 'without_resnet18',
    # 'without_resnet34',
    # 'without_resnet50', 
    # 'without_resnet101',
    # 'without_resnet152'
    # 'cnn'
]
all_train_losses = []
all_val_accuracies = []
for model_name in model_configs:
    print(f"Training {model_name}...")
    loss_acc_path = 'pkl/'+model_name+'.pkl'
    if os.path.exists(loss_acc_path):
        with open(loss_acc_path, 'rb') as f:
            data_set = pickle.load(f)
            train_losses = data_set['loss']
            val_accuracies = data_set['acc']
    else:
        train_losses, val_accuracies = test(model_name)
        data_set={
            'loss':train_losses,
            'acc':val_accuracies
        }
        with open(loss_acc_path, 'wb') as f:
            pickle.dump(data_set, f)
    
    all_train_losses.append(train_losses)
    all_val_accuracies.append(val_accuracies)
writer.close()

plot_compare_accuracy(model_configs, all_val_accuracies)
plot_compare_loss(model_configs, all_train_losses)
# plot_compare_accuracy_and_loss(model_configs, all_val_accuracies, all_train_losses)