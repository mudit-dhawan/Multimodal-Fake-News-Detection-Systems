import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms


class TextEncoder(nn.Module):
    def __init__(self, pre_trained_embed, vocab_size, embedding_size, text_enc_dim, num_layers, hidden_size, bidirectional):
        super(TextEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pre_trained_embed)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.text_encoder = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True,
                               batch_first=True)

        self.text_enc_fc = torch.nn.Linear(self.hidden_size*self.hidden_factor, text_enc_dim)

    def forward(self, x):
        
        x = self.embedding(x)
        
#         print("emb ", x.shape)

        _, (hidden, _not) = self.text_encoder(x)
        
#         print("encoding hidden", hidden.shape)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(x.shape[0], self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        
#         print("encoding hidden", hidden.shape)
        
        x = self.text_enc_fc(hidden)

        return x

class VisualEncoder(nn.Module):
    def __init__(self, enc_img_dim, img_fc1_out, img_fc2_out):
        super(VisualEncoder, self).__init__()
        
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(enc_img_dim, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)
    
    def forward(self, x):

        x_cnn = self.vis_encoder(x)

        x = self.vis_enc_fc1(x_cnn)

        x = self.vis_enc_fc2(x)

        return x, x_cnn

class VisualDecoder(nn.Module):
    def __init__(self, latent_dim, dec_fc_img_1, decoded_img):
        super(VisualDecoder, self).__init__()

        self.vis_dec_fc1 = nn.Linear(latent_dim, dec_fc_img_1)

        self.vis_dec_fc2 = nn.Linear(dec_fc_img_1, decoded_img)
    
    def forward(self, x):

        x = self.vis_dec_fc1(x)

        x = self.vis_dec_fc2(x)

        return x


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, latent_size, hidden_size, num_layers, bidirectional):
        super(TextDecoder, self).__init__()

        self.max_len = max_len
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.text_decoder = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True)

        self.latent2hidden = nn.Linear(latent_size, hidden_size )  ## dec text fc 

        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
    
    def forward(self, x, max_length):

        hidden = self.latent2hidden(x)
#         print("hidden shgape ",hidden.shape)
#         print("max len ", max_length)

#         if self.bidirectional or self.num_layers > 1:
#             # unflatten hidden state
#             hidden = hidden.view(self.hidden_factor, x.shape[0], self.hidden_size)
#         else:
#             hidden = hidden.unsqueeze(0)
        
#         print("hidden shgape ",hidden.shape)
        
#         hidden = hidden.unsqueeze(1)
        
#         print("hidden shgape unsqueezed",hidden.shape)
        
        repeat_hidden = hidden.unsqueeze(1).repeat(1, max_length, 1)  ## repeat the hidden input to the max_len

        # decoder forward pass
        outputs, _ = self.text_decoder(repeat_hidden)
        
        outputs = outputs.contiguous()
#         print("outputs shape after lstm ", outputs.shape)

        b,s,_ = outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=1)
#         logp = nn.functional.log_softmax(self.outputs2vocab(outputs), dim=-1)
#         print("logp shape before ", logp.shape)
        logp = logp.view(b, s, self.vocab_size)
#         print("logp shape after ", logp.shape)
        
        return logp


class MVAE(nn.Module):

    def __init__(self, params_dict):
        super(MVAE, self).__init__()

        self.text_encoder = TextEncoder(params_dict['pre_trained_embed'], params_dict['vocab_size'], params_dict['embedding_size'], params_dict['text_enc_dim'], params_dict['num_layers'], params_dict['hidden_size'], params_dict['bidirectional'])

        self.visual_encoder = VisualEncoder(params_dict['enc_img_dim'], params_dict['img_fc1_out'], params_dict['img_fc2_out'])

        self.text_decoder = TextDecoder(params_dict['vocab_size'], params_dict['embedding_size'], params_dict['max_len'], params_dict['latent_dim'], params_dict['hidden_size'], params_dict['num_layers'], params_dict['bidirectional'])

        self.visual_decoder = VisualDecoder(params_dict['latent_dim'], params_dict['dec_fc_img_1'], params_dict['enc_img_dim'])

        self.combined_fc = torch.nn.Linear((params_dict['text_enc_dim'] + params_dict['img_fc2_out']), params_dict['combined_fc_out'])

        self.fc_mu = nn.Linear(params_dict['combined_fc_out'], params_dict['latent_dim'])
        self.fc_var = nn.Linear(params_dict['combined_fc_out'], params_dict['latent_dim'])

        


        self.fnd_module = nn.Sequential(
                            nn.Linear(params_dict['latent_dim'], params_dict['fnd_fc1']),
                            nn.Tanh(),
                            nn.Linear(params_dict['fnd_fc1'], params_dict['fnd_fc2']),
                            nn.Tanh(),
                            nn.Linear(params_dict['fnd_fc2'], 1),
                            nn.Sigmoid()
        )

        

    def encode(self, text_ip, img_ip):
        encoded_text = self.text_encoder(text_ip)

        encoded_img, cnn_enc_img = self.visual_encoder(img_ip)

        combined = torch.cat(
            [encoded_text, encoded_img], dim=1
        )

        result = self.combined_fc(combined)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return cnn_enc_img, mu, log_var


    def decode(self, z, max_len):
        recon_text = self.text_decoder(z, max_len)

        recon_img = self.visual_decoder(z)

        return [recon_text, recon_img]
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
    def forward(self, text_ip, img_ip):

        ## encoder network 
        cnn_enc_img, mu, log_var = self.encode(text_ip, img_ip)

        z = self.reparameterize(mu, log_var)
        
#         print("text ip shape",text_ip.shape)

        recon_text, recon_img = self.decode(z, text_ip.shape[1])

        fnd_out = self.fnd_module(z)

        return  [fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img]