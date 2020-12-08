import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from models import *



def initialize_model(parameter_dict_model, parameter_dict_opt, device, train_dataloader, epochs=4):
    """Initialize the Fused model, the optimizer and the learning rate scheduler.
    @params
    """
    # Instantiate Models 
    vision_module = Vision_Module(parameter_dict_model['encoded_image_size']) # images 
    language_module = Text_Module(parameter_dict_model['freeze_bert']) # text
    attention_module = CrossAttentionModule(parameter_dict_model['text_enc_dim'], parameter_dict_model['img_enc_dim'], parameter_dict_model['att_dim']) # cross-attention module 
    
    # Tell PyTorch to run the model on GPU
    vision_module.to(device)    

    language_module.to(device)    
    
    attention_module.to(device)
    
    # Instantiate multi-modal pattern
    fused_model = LanguageAndVisionConcat(
            num_classes=parameter_dict_model['nb_classes'],
            language_module=language_module,
            vision_module=vision_module,
            attention_module=attention_module,
            attention_dim=parameter_dict_model['att_dim'],
            language_feature_dim=parameter_dict_model['text_enc_dim'],
            vision_feature_dim=parameter_dict_model['vision_feature_dim'],
            fusion_output_size=parameter_dict_model['fusion_output_size'],
            dropout_p=parameter_dict_model['dropout_p'])
    
    fused_model.to(device)
    
    # Create the optimizer
    optimizer = AdamW(fused_model.parameters(),
                      lr=parameter_dict_opt['l_r'],
                      eps=parameter_dict_opt['eps'])

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return fused_model, optimizer, scheduler