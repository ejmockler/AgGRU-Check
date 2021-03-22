# hyperparameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from model import GRU

from ray.tune.integration.torch import DistributedTrainableCreator, distributed_checkpoint_dir, is_distributed_trainable

# set processing device
import torch
import torch.nn as nn
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# data 

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.optim as optim
from ray.tune import CLIReporter
import os

import numpy as np
from functools import partial

from loadData import declareFields

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Tokenize peptide sequences by splitting into individual amino acids
def split(sequence):
    return [char for char in sequence] 

def load_data(fields, root_dir="./"):
    # Create simple TabularDatasets from train/test CSV
    trainData, testData = TabularDataset.splits(path=root_dir, train="train.csv", test="test.csv", format='CSV', fields=fields, skip_header=True)
    return trainData, testData

def optimizeHyperparameters(config, data_dir, checkpoint_dir=None, out_dir='./output', amyloid=True):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    fullDataset, fields, sequence = declareFields(data_dir=data_dir)
    trainData, testData = load_data(fields, data_dir)

    # create iterators for current fold
        # sort by sequence length to keep batches consistent 
    train_iter =  BucketIterator(trainData, batch_size=config['batch_size'], sort_key=lambda x: len(x.sequence),
                        device=device, sort=True, sort_within_batch=True)
    test_iter =  BucketIterator(testData, batch_size=config['batch_size'], sort_key=lambda x: len(x.sequence),
                        device=device, sort=True, sort_within_batch=True)
    eval_every=len(train_iter) // 2

    # initialize model
    model = GRU(vocab=sequence.vocab, dimension=config['dimension'], sequenceDepth=config['sequence_feature_depth'], dropoutWithinLayers=config['dropout_within_layers'], dropoutOutput=config['dropout_output'])

    # load optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.SmoothL1Loss()

    # training loop
    model.train()
    for epoch in range(config['number_of_epochs']):
        for (header, (sequence, sequence_len), prionLabel, amyloidLabel), _ in train_iter:           
            prionLabel = prionLabel.to(device)
            amyloidLabel = amyloidLabel.to(device)
            sequence = sequence.to(device)
            sequence_len = sequence_len.to(device)
            model.to(device)
            output = model(sequence, sequence_len.cpu())

            amyloidLoss = criterion(output, amyloidLabel)
            prionLoss = criterion(output, prionLabel)
            loss = amyloidLoss if amyloid else prionLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1
            
            # evaluation for this epoch
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                    # validation loop
                    for (header, (sequence, sequence_len), prionLabel, amyloidLabel), _ in test_iter:   
                        prionLabel = prionLabel.to(device)
                        amyloidLabel = amyloidLabel.to(device)
                        sequence = sequence.to(device)
                        sequence_len = sequence_len.to(device)
                        output = model(sequence, sequence_len.cpu())
                        
                        amyloidLoss = criterion(output, amyloidLabel)
                        prionLoss = criterion(output, prionLabel)
                        loss = amyloidLoss if amyloid else prionLoss
                        valid_running_loss += loss.item()
            
                # record loss
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(test_iter)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                with  tune.checkpoint_dir(epoch) as file_path:
                    path = os.path.join(file_path, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                # resetting running values
                valid_running_loss = 0.0
                running_loss = 0.0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                        .format(epoch+1, config['number_of_epochs'], global_step, config['number_of_epochs']*len(train_iter),
                                average_train_loss, average_valid_loss))
                tune.report(valid_loss=average_valid_loss)
                save_metrics(file_path + f'/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.path.abspath("./")    
    fullDataset, fields, sequence = declareFields(data_dir=data_dir)
    
    config = {
        "number_of_epochs": tune.sample_from(lambda _: np.random.randint(2, 25)),
        "dropout_within_layers": tune.sample_from(lambda _: np.random.uniform(high = 0.5)),
        "dropout_output": tune.sample_from(lambda _: np.random.uniform(high = 0.5)),
        "sequence_feature_depth": tune.sample_from(lambda _: np.random.randint(16, 192)),
        "dimension": tune.sample_from(lambda _: np.random.randint(16, 192)),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
        "num_gpu": 1,
        "num_workers": 2
    }
    scheduler = ASHAScheduler(
        metric="valid_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["valid_loss", "training_iteration"])
    result = tune.run(
        partial(optimizeHyperparameters, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["valid_loss"]))

    best_trained_model = GRU(vocab=sequence.vocab, dimension=best_trial.config['dimension'], sequenceDepth=best_trial.config['sequence_feature_depth'], dropoutWithinLayers=best_trial.config['dropout_within_layers'], dropoutOutput=best_trial.config['dropout_output'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value

    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=30, max_num_epochs=10, gpus_per_trial=1)

    print('Finished Optimizing Hyperparameters!')