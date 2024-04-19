import os
import yaml
import pandas as pd

from git import Repo
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

import sentencepiece as spm

import torch.utils
import torch.nn as nn
import torch
import torch.optim as optim
import torchtext

import dataloaders
import models
import test_torch_and_cuda
import utils
import socket
import getpass
import datetime

import wandb

def train_one_epoch(model,
                    tokenizer,
                    criterion,
                    optimizer,
                    data_loader,
                    preprocessing,
                    device):
    epoch_loss = 0
    n_batches = 0
    #iterations = tqdm(data_loader)
    model.train()
    for xy in tqdm(data_loader):
        x, y = preprocessing(xy)
        n_batches += 1
        x = x.to(device)
        y = y.to(device).reshape( (y.shape[0] * y.shape[1],))
        y_est, _ = model(x)
        y_est = y_est.reshape( (y_est.shape[0] * y_est.shape[1], -1 ))
        #print(y.shape, y_est.shape)
        loss = criterion(y_est, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / n_batches

def validate(model,
         tokenizer,
         criterion,
         data_loader,
         preprocessing,
         device):
    
    model.eval()
    eval_loss = 0
    n_batches = 0
    with torch.no_grad():
        for xy in tqdm(data_loader):
            x, y = preprocessing(xy)
            n_batches += 1
            x = x.to(device)
            y = y.to(device).reshape( (y.shape[0] * y.shape[1],))
            y_est, _ = model(x)
            y_est = y_est.reshape( (y_est.shape[0] * y_est.shape[1], -1 ))
            
            loss = criterion(y_est, y)
            # Accumulate loss
            eval_loss += loss.item()
            
    return eval_loss / n_batches
    

def train(model,
          tokenizer,
          criterion,
          optimizer,
          scheduler,
          data_loader_train,
          data_loader_val,
          preprocessing,
          num_epochs,
          early_stop_patience,
          ):
    
    device = next(model.parameters()).device
    es = utils.EarlyStop(early_stop_patience)
    best_val_loss = 9999999
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model,
                    tokenizer,
                    criterion,
                    optimizer,
                    data_loader_train,
                    preprocessing,
                    device)
        val_loss = validate(model,
                    tokenizer,
                    criterion,
                    data_loader_val,
                    preprocessing,
                    device
                    )
        scheduler.step(val_loss)
        print(train_loss, val_loss)
        wandb.log({"train_loss": train_loss, "val_loss" : val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'sandbox/model.pth')
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Val_loss: {val_loss}')
        
        if es(val_loss):
            print("Exiting training loop due to early stopping")
            break
        
    #artifact = wandb.Artifact('model', type='model')
    #artifact.add_file('sandbox/model.pth')
    #wandb.log_artifact(artifact)    

def main():
    # Initializing
    print("Initializing train script")
    print("PID: " + str(os.getpid()))
    print("Machine: " + socket.gethostname())
    print("User: " + getpass.getuser())
    print("Timestamp: " + str(datetime.datetime.now()))
    
    settings = {
        'PID' : str(os.getpid()),
        'Machine' : socket.gethostname(),
        'User' : getpass.getuser(),
        'Timestamp' : str(datetime.datetime.now())
    }
    
    # Load configurations
    cfg = yaml.safe_load(open('config/experiment.yml', 'r'))
    cfg |= settings
    
    # Check state
    test_torch_and_cuda.print_cuda_diagnostics()
    
    print("Loading tokenizer")
    # Load model
    if not os.path.exists('sandbox/' + cfg['config']['sp_tokenizer_prefix'] + '.model'):
        spm.SentencePieceTrainer.train(input=cfg['config']['dataset_path'],
                                model_prefix='sandbox/' + cfg['config']['sp_tokenizer_prefix'],
                                vocab_size=cfg['config']['vocabulary_size'],
                                user_defined_symbols=['<UNK>', '<SOS>', '<EOS>', '<s>', '</s>'],
                            )
        
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('sandbox/' + cfg['config']['sp_tokenizer_prefix']+ '.model')
    
    print("Loading model")
    model = models.TextPredictorRNN(vocabulary_size=cfg['config']['vocabulary_size'],
                                    n_latent=cfg['config']['n_latent'],
                                    num_layers=cfg['config']['num_layers'],
                                    embedding_dim=cfg['config']['embedding_dim']).cuda()

    print("Loading dataset and dataloaders")    
    # Load dataset / dataloaders
    print("... reading data...")
    texts = list(pd.read_csv(cfg['config']['dataset_path'])['review'])
    X_train, X_test = train_test_split(texts, test_size=0.20, random_state=42)
    
    print("... pre-processing data... ")
    
    dataset_train = dataloaders.DatasetFromList(X_train)
    dataset_val = dataloaders.DatasetFromList(X_test)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg['config']['batch_size'],
        shuffle=True,
        num_workers=cfg['config']['num_workers'])

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg['config']['batch_size'],
        shuffle=True,
        num_workers=cfg['config']['num_workers'])

    preprocessing = nn.Sequential(
        models.StringsToInt(tokenizer),
        models.FixedSeq(cfg['config']['seq_len'], 0),
        torchtext.transforms.ToTensor(),    
        models.GetLast(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['config']['lr'], weight_decay=cfg['config']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['config']['reduce_lr_ratio'], patience=cfg['config']['reduce_lr_patience'])

    print("Performing a sanity check on the model...")
    works = utils.sanity_check(model, tokenizer, dataset_train, preprocessing)
    if not works:
        print("Model does not work. Exiting...")
        exit()

    print("Model works")        

    # Checking if you forgot to commit code before running test
    print("Checking for uncommitted changes and untracked files")
    if not utils.git_check_if_commited():
        print("Solve git problems and then try again!")
        exit()

    # Train
    print("Entering train...")
    run = wandb.init(
        project=cfg['project'],
        notes="",
        tags=["experimental", "rnn"],
        config=cfg['config'],
        )
    try:
        train(model,
            tokenizer,
            criterion,
            optimizer,
            scheduler,
            data_loader_train, 
            data_loader_val,
            preprocessing,
            cfg['config']['num_epochs'],
            cfg['config']['early_stopping_patience']
            )
        
    except KeyboardInterrupt:
        print("Trying to exit gracefully...")
    
    wandb.finish()
    
if __name__ == "__main__":
    main()