import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
import random
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import torchtext.vocab as vocab
from torchtext.data import Field

from gensim.models import Word2Vec


spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self, freq_threshold, embed_size):
        self.itos = {0: "<PAD>"}
        self.stoi = {"<PAD>": 0}
        self.freq_threshold = freq_threshold
        self.embed_size = embed_size
        self.pre_trained_embed = None

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_w2v(self, sentence_list):
        sentences = [self.tokenizer_eng(sentence) for sentence in sentence_list]
        # train model
        model = Word2Vec(sentences, min_count=self.freq_threshold, size=self.embed_size)  

#         print(model.wv.vocab)  

        model.save('embeddings.txt')

        TEXT = Field()
        TEXT.build_vocab(sentences, min_freq=self.freq_threshold)

        w2v_vec = []
        for token, idx in self.stoi.items():
            if token in model.wv.vocab.keys():
                w2v_vec.append(torch.FloatTensor(model.wv[token]))
            else:
                w2v_vec.append(torch.zeros(self.embed_size))

        TEXT.vocab.set_vectors(self.stoi, w2v_vec, self.embed_size)

        self.pre_trained_embed = torch.FloatTensor(TEXT.vocab.vectors)
        

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 1

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else random.choice(list(self.stoi.values()))
            for token in tokenized_text
        ]


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, root_dir, image_transform, vocab=None, freq_threshold=5, embed_size=32):
        """
        Args:
            csv_file (string): Path to the csv file with text and img name.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.vocab = vocab

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.txt = self.df["content"]
        self.label = self.df['label']
        
        if self.vocab == None:
        # Initialize vocabulary and build vocab
            self.vocab = Vocabulary(freq_threshold,  embed_size)
            self.vocab.build_vocabulary(self.txt.tolist())

            self.vocab.build_w2v(self.txt.tolist())

    def __len__(self):
        return len(self.df)
    
    def pre_processing_text(self, sent):
        pass
        
    def __getitem__(self, idx):
        
        img_name = self.root_dir + self.imgs[idx]

        image = Image.open(img_name).convert("RGB")
        
        image = self.image_transform(image)
        
        text = self.txt[idx]
        
        # numericalized_text = [self.vocab.stoi["<SOS>"]]
        # numericalized_text += self.vocab.numericalize(text)
        numericalized_text = self.vocab.numericalize(text)
        # numericalized_text.append(self.vocab.stoi["<EOS>"])


        label = self.label[idx]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        sample = {'image': image, 'text': torch.tensor(numericalized_text), 'label': label}

        return sample


class MyCollate:
    def __init__(self, pad_idx, max_len):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __call__(self, batch):
        
#         print(batch)
        imgs = [item['image'].unsqueeze(0) for item in batch]
        
        labels = [item['label'].unsqueeze(0) for item in batch]
        text = [item['text'] if item['text'].shape[0] <= self.max_len else item['text'][:self.max_len] for item in batch]
    
#         batch_imgs, batch_text, batch_labels = batch['image'], batch['text'], batch['label']

#         imgs = [item.unsqueeze(0) for item in batch_imgs]
        imgs = torch.cat(imgs, dim=0)

#         labels = [item.unsqueeze(0) for item in batch_labels]
        labels = torch.cat(labels, dim=0)

#         text = [item for item in batch_text]
        text = pad_sequence(text, batch_first=True, padding_value=self.pad_idx)


        sample = {'image': imgs, 'text': text, 'label': labels}

        return sample


def get_loader(
    df,
    root_dir,
    image_transform,
    vocab=None,
    batch_size=8,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FakeNewsDataset(df, root_dir, image_transform, vocab=None, freq_threshold=6, embed_size=32)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
#         num_workers=num_workers,
        shuffle=shuffle,
#         pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, max_len=500),
    )

    return loader, dataset