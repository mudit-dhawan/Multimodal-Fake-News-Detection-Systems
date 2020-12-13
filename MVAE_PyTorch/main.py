import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math
from torch.utils.tensorboard import SummaryWriter

from data_loader import * 
from models import * 
from train_val import * 

df_train = pd.read_csv("../datasets/AAAI_dataset/gossip_train.csv")
df_test = pd.read_csv("../datasets/AAAI_dataset/gossip_test.csv")

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

    # define a callable image_transform with Compose
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

root_dir = "../datasets/AAAI_dataset/Images/"

# Run function `preprocessing_for_bert` on the dataset
train_dataloader, transformed_dataset_train = get_loader(df=df_train, root_dir=root_dir+"gossip_train/", image_transform=image_transform, vocab=None)

test_dataloader, transformed_dataset_test = get_loader(df=df_test, root_dir=root_dir+"gossip_test/", image_transform=image_transform, vocab=transformed_dataset_train.vocab)

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

fnd_loss_fn = nn.BCELoss()
recon_text_loss = nn.NLLLoss()

def loss_function(ip_text, ip_img, ip_label, mu, log_var, rec_text, rec_img, fnd_label, lambda_wts):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """

    fnd_loss = fnd_loss_fn(fnd_label, ip_label)

    recons_loss =F.mse_loss(ip_img, rec_img)
    
    rec_text = rec_text.view(-1, rec_text.size(2))
    
    ip_text = ip_text.view(-1)

    text_loss = recon_text_loss(rec_text, ip_text)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = lambda_wts['fnd'] * fnd_loss + lambda_wts['img'] * recons_loss + lambda_wts['kld'] * kld_loss + lambda_wts['text'] * text_loss

    return loss

params_dict_model = {
    'pre_trained_embed': transformed_dataset_train.vocab.pre_trained_embed,
    'latent_dim': 32,
    'combined_fc_out': 64,
    'dec_fc_img_1': 1024,
    'enc_img_dim': 4096,
    'vocab_size': len(transformed_dataset_train.vocab.stoi),
    'embedding_size': 32,
    'max_len': 20,
    'text_enc_dim': 32,
    'latent_size': 32,
    'hidden_size': 32,
    'num_layers': 1,
    'bidirectional': True,
    'img_fc1_out': 1024,
    'img_fc2_out': 32,
    'fnd_fc1': 64,
    'fnd_fc2': 32
}

parameter_dict_opt={'l_r': 3e-5,
                    'eps': 1e-8
                    }

EPOCHS = 1

set_seed(42)    # Set seed for reproducibility


final_model = MVAE(params_dict_model)
final_model.to(device)

# Create the optimizer
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

# Total number of training steps
total_steps = len(train_dataloader) * EPOCHS

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value
                                            num_training_steps=total_steps)



## Instantiate the tensorboard summary writer
writer = SummaryWriter('runs/mvae_exp1')

train(model=final_model, loss_fn=loss_function, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=test_dataloader, epochs=EPOCHS, evaluation=True, device=device, param_dict_model=params_dict_model, param_dict_opt=parameter_dict_opt, save_best=True, file_path='./saved_models/mvae_model.pt', writer=writer)