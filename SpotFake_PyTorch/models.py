import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel


# Create the Bert custom class 
class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=32, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True, 
                    return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        ## odict_keys(['last_hidden_state', 'pooler_output', 'attentions'])
        ## last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) 

        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc(out['last_hidden_state']))
        )    

        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        ) 
        
        return x
    
    def fine_tune(self):
        """
        keep the weights fixed or not  
        """
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module
            

            
class VisionEncoder(nn.Module):
    """Visual Feature extraction
    """
    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        """

        """
        super(VisionEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        

        
    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        x = self.vis_encoder(x)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        return x
    
    def fine_tune(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module
                
class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        model_params
        
    ):
        super(LanguageAndVisionConcat, self).__init__()
        
        self.language_module = TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'], model_params['dropout_p'], model_params['fine_tune_text_module'])
        self.vision_module = VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'], model_params['dropout_p'], model_params['fine_tune_vis_module'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out'] + model_params['img_fc2_out']), 
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'], 
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])


    def forward(self, text, image, label=None):

        ## Pass the text input to Bert encoder 
        text_features = self.language_module(text[0], text[1])

        ## Pass the image input 
        image_features = self.vision_module(image)

        ## concatenating Image and text 
        combined_features = torch.cat(
            [text_features, image_features], dim=1
        )        

        combined_features = self.dropout(combined_features)
        
        fused = self.dropout(
            torch.nn.functional.relu(
            self.fusion(combined_features)
            )
        )
        
        prediction = torch.nn.functional.sigmoid(self.fc(fused))

        return prediction      
