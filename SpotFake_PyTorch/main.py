import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math
from torch.utils.tensorboard import SummaryWriter

from models import *
from data_loader import *
from train_val import *


df_train = pd.read_csv("./datasets/AAAI_dataset/gossip_train.csv")
df_test = pd.read_csv("./datasets/AAAI_dataset/gossip_test.csv")

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

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Specify `MAX_LEN`
MAX_LEN = 500
root_dir = "./datasets/AAAI_dataset/Images/"

# Run function `preprocessing_for_bert` on the dataset
transformed_dataset_train = FakeNewsDataset(df_train, root_dir+"gossip_train/", image_transform, tokenizer, MAX_LEN)

transformed_dataset_val = FakeNewsDataset(df_test, root_dir+"gossip_test/", image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


# Specify loss function
loss_fn = nn.NLLoss()


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

parameter_dict_model={
    'text_fc2_out': 32, 
    'text_fc1_out': 2742, 
    'dropout_p': 0.4, 
    'fine_tune_text_module': False,
    'img_fc1_out': 2742, 
    'img_fc2_out': 32, 
    'dropout_p': 0.4, 
    'fine_tune_vis_module': False,
    'fusion_output_size': 35
}

parameter_dict_opt={'l_r': 3e-5,
                    'eps': 1e-8
                    }


EPOCHS=50

set_seed(7)    # Set seed for reproducibility

final_model = LanguageAndVisionConcat(parameter_dict_model)

final_model = final_model.to(device) 

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
writer = SummaryWriter('runs/multi_att_exp3')


train(model=final_model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=150, evaluation=True, device=device, param_dict_model=parameter_dict_model, param_dict_opt=parameter_dict_opt, save_best=True, file_path='./saved_models/best_model.pt', writer=writer)