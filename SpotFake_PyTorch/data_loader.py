import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import torch.nn.functional as F
from transformers import BertModel
import random
import time
import os
import re


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        """
        Args:
            csv_file (string): Path to the csv file with text and img name.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]
    
    def pre_processing_BERT(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []
        
        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
     
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.root_dir + self.csv_data['image'][idx]
        
#         image = io.imread(img_name)
        image = Image.open(img_name).convert("RGB")
        
        image = self.image_transform(image)
        
        text = self.csv_data['content'][idx]
        
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        label = self.csv_data['label'][idx]
        label = torch.tensor(label)

        sample = {'image': image, 'BERT_ip': [tensor_input_id, tensor_input_mask], 'label':label}
        # print(idx)

        return sample