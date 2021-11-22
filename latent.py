import torch
from pytorch_pretrained_biggan import  truncated_noise_sample
from pre_trained_transformers_tf import get_captions
from gpt2.encoder import get_encoder
import time
import os
import random
from itertools import cycle

class DeepMindBigGANLatentSpace(torch.nn.Module):
    def __init__(self, config):
        super(DeepMindBigGANLatentSpace, self).__init__()
        self.config = config

        self.z = torch.nn.Parameter(torch.tensor(truncated_noise_sample(self.config.batch_size)).to(self.config.device))
        self.class_labels = torch.nn.Parameter(torch.rand(self.config.batch_size, self.config.num_classes).to(self.config.device))
    
    def set_values(self, z, class_labels):
        self.z.data = z
        self.class_labels.data = class_labels

    def set_from_population(self, x):
        self.z.data = torch.tensor(x[:,:self.config.dim_z].astype(float)).float().to(self.config.device)
        self.class_labels.data = torch.tensor(x[:,self.config.dim_z:].astype(float)).float().to(self.config.device)

    def forward(self):
        z = torch.clip(self.z, -2, 2)
        class_labels = torch.softmax(self.class_labels, dim=1)

        return z, class_labels
        

class StyleGAN2LatentSpace(torch.nn.Module):
    def __init__(self, config):
        super(StyleGAN2LatentSpace, self).__init__()
        self.config = config

        self.z = torch.nn.Parameter(torch.randn(self.config.batch_size, self.config.dim_z).to(self.config.device))
    
    def set_values(self, z):
        self.z.data = z

    def set_from_population(self, x):
        self.z.data = torch.tensor(x.astype(float)).float().to(self.config.device)

    def forward(self):
        return (self.z, )


class GPT2LatentSpace(torch.nn.Module):
    def __init__(self, config):
        super(GPT2LatentSpace, self).__init__()
        self.config = config

        if os.path.isfile(self.config.target + '.pt'):
            self.z = torch.load(self.config.target + '.pt')
        else:
            self.enc = get_encoder(config)
            ini_t = time.time()
            captions = get_captions(self.config.target)
            end_t = time.time()
            print('[INFO] Generated captions. Time: {}'.format(end_t - ini_t))
            for caption in captions:
                print('Caption: ', caption)
            captions_tokenized = [ self.enc.encode(caption) for caption in captions ]

            vecs = []
            for vec in cycle(captions_tokenized):
                if len(vecs) >= self.config.batch_size:
                    break
                else:
                    new_vec = vec
                    while len(new_vec) < self.config.dim_z:
                        new_vec.append(random.choice(vec))
                    vecs.append(new_vec)
            self.z = torch.as_tensor(vecs)
            torch.save(self.z, self.config.target + '.pt')

        #self.z = torch.randint(0, self.config.encoder_size, size=(self.config.batch_size, self.config.dim_z)).to(self.config.device)
        #self.z = torch.zeros(self.config.batch_size, self.config.dim_z)
    
    def set_values(self, z):
        self.z.data = z

    def set_from_population(self, x):
        self.z.data = torch.tensor(x.astype(int)).long().to(self.config.device)

    def forward(self):
        return (self.z, )