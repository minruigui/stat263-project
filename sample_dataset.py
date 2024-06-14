import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid


# Model Hyperparameters

dataset_path = './criteo/train.csv'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 100

x_dim  = 784
hidden_dim = 400


lr = 1e-5

epochs = 10


import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


df = pd.read_csv(dataset_path)
print(df.head())


# Assuming 'df' is your DataFrame
columns_to_keep = [col for col in df.columns if col.startswith('C')]
columns_to_keep.append("label")
filtered_df = df[columns_to_keep]

# Print the shape of the filtered DataFrame
print(filtered_df.shape)
# Print the columns of the filtered DataFrame
print(filtered_df.columns)
# Print the number of unique values in each column of the filtered DataFrame
num_classes = filtered_df.nunique().sort_values().to_dict()
print(num_classes)
print(filtered_df.max())



"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.LayerNorm1 = nn.LayerNorm(hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, latent_dim*2)
        self.LayerNorm2 = nn.LayerNorm( latent_dim*2)
        self.FC_mean = nn.Linear( latent_dim*2, latent_dim)
        self.FC_var = nn.Linear( latent_dim*2, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.LayerNorm1(self.FC_input(x)))
        h = self.LeakyReLU(self.LayerNorm2(self.FC_input2(h)))

        
        return h

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.LayerNorm1 = nn.LayerNorm(hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm2 = nn.LayerNorm(hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.LayerNorm1(self.FC_hidden(x)))
        h = self.LeakyReLU(self.LayerNorm2(self.FC_hidden2(h)))
        x_hat = torch.nn.functional.log_softmax(self.FC_output(h), dim=1)
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z
        
    def encode(self, x):
        h = self.Encoder(x)
        return h
    
    def decode(self, z):
        x_hat = self.Decoder(z)
        return x_hat
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

class ModelsMap(nn.Module):
    def __init__(self, columns_to_keep, num_classes, hidden_dim, latent_dim, device):
        super(ModelsMap, self).__init__()
        self.models = nn.ModuleDict()
        self.num_classes = []
        self.latent_num_classes = []
        self.activation = nn.ReLU()
        for col in columns_to_keep:
            self.num_classes.append(num_classes[col])
            print(f"{col} has {num_classes[col]} classes")
            self.latent_num_classes.append(latent_dim)
            encoder = Encoder(input_dim=num_classes[col], hidden_dim=hidden_dim, latent_dim=latent_dim)
            decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=num_classes[col])
            model = Model(Encoder=encoder, Decoder=decoder).to(device)
            self.models[col] = model
        self.latent_size = len(columns_to_keep) * latent_dim
        self.mean_projector = nn.Linear(self.latent_size*2, self.latent_size*3).to(device)

        self.mean_projector_1 = nn.Linear(self.latent_size*3, self.latent_size).to(device)

    def forward(self, x):
        hs = []
        for k, v in x.items():
            h = self.models[k].encode(v)
            hs.append(h)
        hs_ = self.mean_projector(torch.cat(hs, dim=1))
        hs_ = self.activation(hs_)
        hs_=self.mean_projector_1(hs_)
        hs_ = torch.split(hs_, self.latent_num_classes, dim=1)
        output = {}
        for idx, (k, v) in enumerate(x.items()):
            mean = self.models[k].Encoder.FC_mean(hs[idx])
            # var = self.models[k].Encoder.FC_var(hs[idx])
            # z = self.models[k].reparameterization(mean, torch.exp(0.5 * var))
            z=mean
            x_hat = self.models[k].decode(z)
            output[k] = [x_hat, 0, 0]
        return output
    def encode(self,x):
        hs = []
        for k, v in x.items():
            h = self.models[k].encode(v)
            hs.append(h)
        hs_ = self.mean_projector(torch.cat(hs, dim=1))
        hs_ = self.activation(hs_)
        hs_=self.mean_projector_1(hs_)
        hs_ = torch.split(hs_, self.latent_num_classes, dim=1)
        output = {}
        for idx, (k, v) in enumerate(x.items()):
            mean = self.models[k].Encoder.FC_mean(hs[idx])
            var = self.models[k].Encoder.FC_var(hs[idx])
            z = self.models[k].reparameterization(mean, torch.exp(0.5 * var))
            output[k] = [z, mean, var]
        return output
columns_to_keep = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
       'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
       'C22', 'C23', 'C24', 'C25', 'C26',"label"]
model = ModelsMap(columns_to_keep, num_classes, hidden_dim, 4, DEVICE)         
model.load_state_dict(torch.load('nov_model_weights.pth'))
from datasets import load_dataset

# Load the criteox_1 dataset
dataset = load_dataset('./criteo')
test_dataset = dataset["train"]




import torch
# create a dict with default values as {}
from collections import defaultdict
c_dict = defaultdict(dict)
for col in columns_to_keep:
    for i,v in enumerate(set(df[col])):
        c_dict[col][v] = i

# print(c_dict)
import json
with open('c_dict.json', 'w') as fp:
    json.dump(c_dict, fp)
print([num_classes[col] for col in columns_to_keep])
model.eval()

def convert_to(x):
    for col in columns_to_keep:
        x[col] = model(c_dict[col][x[col]])

    
C5_test_dataset = test_dataset.map(convert_to)







print("Finish!!")
