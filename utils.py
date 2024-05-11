import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import auc, roc_curve
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split

str_to_int = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}
int_to_str = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad'}

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.data = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx, 0]
        class_label = int(self.data.iloc[idx, 1])
        class_str = int_to_str[class_label]
        img_name = os.path.join(self.root_dir, class_str, image_filename)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(class_label, dtype=torch.long)
        return image, label

def load_data(batch_size):
    data_transformer = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor()
    ])

    data_dir = './data'
    csv_file = pd.read_csv('./data/labels.csv')
    csv_file = csv_file.drop(columns=['Unnamed: 0'])

    csv_file['label'] = csv_file['label'].map(str_to_int)

    img_dataset = CustomDataset(data_dir, csv_file, data_transformer)

    dataset_size = len(img_dataset)

    train_ratio = 0.85
    test_ratio = 0.15

    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    val_size = dataset_size - train_size - test_size

    train_dataset, test_dataset = random_split(img_dataset, [train_size, test_size])

    val_ratio = 0.15
    train_size = len(train_dataset)
    val_size = int(val_ratio * train_size)

    train_dataset, val_dataset = random_split(train_dataset, [train_size - val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# 绘制混淆矩阵


def plot_confusion_matrix(model, cm):
    plt.figure(figsize=(8, 8))
    sns.set(style="darkgrid")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('figure/confusion_matrix_'+model+'.jpg', dpi=1000)
    plt.show()
    
def plot_auroc(model, all_labels, all_predictions_pro):
    all_predictions_pro=np.vstack(all_predictions_pro)

    true_labels = np.array(all_labels)
    predicted_probabilities = np.array(all_predictions_pro)

    plt.figure(figsize=(10, 8))
    sns.set(style="darkgrid")
    for class_idx in range(predicted_probabilities.shape[1]):
        class_true_labels = true_labels == class_idx
        class_predicted_probs = predicted_probabilities[:, class_idx]
        fpr, tpr, _ = roc_curve(class_true_labels, class_predicted_probs)
        auroc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {class_idx} (AUROC = {auroc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC Curve for Multi-Class Classification')
    plt.legend(loc="best")
    plt.savefig('figure/AUROC_'+model+'.jpg', dpi=1000)
    plt.show()


def plot_accuracy(model, accuracy):
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(accuracy, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Value Accuracy')
    plt.savefig('figure/value_accuracy_'+model+'.jpg', dpi=1000)
    plt.show()


def plot_loss(model, losses):
    plt.figure()
    sns.set(style="darkgrid")
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig('figure/train_loss_'+model+'.jpg', dpi=1000)
    plt.show()

def plot_compare_accuracy(model_configs, all_val_accuracies):
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    palette = sns.color_palette("husl", n_colors=len(model_configs))

    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '--' if "without" in model_name else '-'
        plt.plot(all_val_accuracies[i], label=model_name+'_Accuracy', color=color, linestyle=line_style)

    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig('figure/value_accuracy_compare.jpg', dpi=1000)
    plt.show()

def plot_compare_loss(model_configs, all_train_losses):
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    palette = sns.color_palette("husl", n_colors=len(model_configs))
    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '--' if "without" in model_name else '-'
        plt.plot(all_train_losses[i], label=model_name+'_Loss', color=color, linestyle=line_style)

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig('figure/train_loss_compare.jpg', dpi=1000)
    plt.show()

def plot_compare_accuracy_and_loss(model_configs, all_val_accuracies, all_train_losses):
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    palette = sns.color_palette("husl", n_colors=len(model_configs))

    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '--'
        
        # 绘制Validation Accuracy
        label_accuracy = model_name + '_accuracy'
        plt.plot(all_val_accuracies[i], label=label_accuracy, color=color, linestyle=line_style)
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    # 创建第二个纵坐标标度
    ax2 = plt.gca().twinx()
    for i, model_name in enumerate(model_configs):
        color = palette[i]
        line_style = '-'
        
        # 绘制Training Loss
        label_loss = model_name + '_loss'
        ax2.plot(all_train_losses[i], label=label_loss, color=color, linestyle=line_style)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.grid(True)
    plt.savefig('figure/accuracy_and_loss_compare.jpg', dpi=500)
    plt.show()