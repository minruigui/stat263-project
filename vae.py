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
from torch.optim import Adam

BCE_loss = nn.NLLLoss( reduction='sum')

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = BCE_loss(x_hat,x)
    # KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    KLD = 0

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)


from datasets import load_dataset

# Load the criteox_1 dataset
dataset = load_dataset('./criteo')
train_dataset = dataset["train"]




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
    
C5_train_dataset = train_dataset.map(lambda x:{col+"_I":c_dict[col][x[col]] for col in columns_to_keep},num_proc=32).select_columns([col+"_I" for col in columns_to_keep])



import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'stat263'
import wandb
wb = wandb.login()
print("Logged: ", wb)


print("Start training VAE...")
if wb:
    wandb.init(
    # Set the project where this run will be logged
    project="stat263", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{str(columns_to_keep)}_no_vae", 
    # Track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "vae",
    "dataset": "criteo_sample",
    "epochs": epochs,
    })
  
from torch.utils.data import DataLoader
model.train()
batch_size=128
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, x in enumerate(DataLoader(C5_train_dataset, batch_size=batch_size)):
        xs = {}
        x_onehots = {}
        for col in columns_to_keep:
            x_=x[col+"_I"].to(DEVICE)
            xs[col]=( x_)
            x_onehot = torch.nn.functional.one_hot(x_,num_classes[col]).float()
            x_onehots[col]=x_onehot

        optimizer.zero_grad()
        total_loss = 0
        losses = {}
        output = model(x_onehots)
        for k,v in output.items():
            x_hat, mean, log_var = v
            x = xs[k]
            loss = loss_function(x, x_hat, mean, log_var)
            total_loss+=loss
            losses[k]=loss
        #calcualte the accuracy of the x_one
        
        overall_loss += total_loss.item()
        
        total_loss.backward()
        optimizer.step()
        if batch_idx%1000 == 0:
            ys={}
            for k,v in output.items():
                x_hat, mean, log_var = v
                y = torch.argmax(x_hat,dim=1)
                x = xs[k]
                ys[k]=(y==x).sum().item()/len(x)
            
            print("Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t ]".format(
                epoch, batch_idx * len(x), len(C5_train_dataset),
                100. * batch_idx / len(C5_train_dataset),
                total_loss.item() / len(x)),ys)
            if wb:
                wandb.log({"loss": total_loss.item() / len(x), "accuracy": ys})
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
if wb:
    wandb.finish()
print("Finish!!")

torch.save(model.state_dict(), 'nov_model_weights.pth')