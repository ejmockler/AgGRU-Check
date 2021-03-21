#!/usr/bin/env python
# coding: utf-8

# # Amyloid & prion protein classification
# 
# Based upon code & knowledge outlined at https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0

# ## Preprocess
# ### Parse FASTA into CSV
# The PLAAC dataset is in FASTA format, which will be parsed into a CSV table for preprocessing. We'll record the gene name, Uniprot sequence range & sequence text, and add positive prion & amyloid classifications.

# In[1]:


import sys, os

plaacFasta = 'positiveExamples/yeastPrion.fasta'
negativeFasta = 'negativeExamples/negative.fasta'
positiveDataset = 'positiveExamples/positiveDataset.csv'
negativeDataset = 'negativeExamples/negativeDataset.csv'
destination_folder = "./output"

def fastaToCSV(fastaPath, csvPath, prionClassification = 0, amyloidClassification = 0):
    # Read in FASTA
    with open(fastaPath, 'r') as file:
        lines = ["header,sequence,prion,amyloid"]
        lines_i = file.readlines()

        seq = ''
        index = 0
        while index < len(lines_i):
            if lines_i[index][0] == '>':
                'Fasta head line'
                # append sequence & prion classification
                if seq: lines.append(seq + ',' + str(prionClassification) + ',' + str(amyloidClassification))
                seq = ''
                # remove FASTA syntactic sugar
                seq_id = lines_i[index].strip().replace(',', '').replace('>', '')
                # remove uniprot metadata using pipe & equal sign delimiter
                seq_id = " ".join((seq_id.split(" "))[1:]) if "|" in seq_id else seq_id
                lastHeaderIndex = seq_id.index("=") - 3 if ('=' in seq_id) else len(seq_id)
                lines.append('\n' + seq_id[:lastHeaderIndex] + ',')
            else:
                'Sequence line'
                seq += lines_i[index].strip()
                if (index == len(lines_i) - 1): lines.append(seq + ',' + str(prionClassification) + ',' + str(amyloidClassification))
            index += 1
        lines.append("\n")
        file.close()

    # Output CSV file
    with open(csvPath, 'w') as file:
        file.writelines(lines)
        file.close()
    return

# negative
fastaToCSV(negativeFasta, negativeDataset)
# PLAAC prions
fastaToCSV(plaacFasta, positiveDataset, 1, 1)


# ### Parse JSON into CSV
# The Amypro dataset is in JSON format, so we'll use JSON tools to parse data into a CSV table of the same format as above. 

# In[2]:


import json 
import csv 

amyproDataset = 'positiveExamples/amyproAmyloids.json'

with open(amyproDataset, encoding = 'utf-8-sig') as json_file: 
    data = json.load(json_file) 
    lines = []
    for amyloid in data:
        lines.append(amyloid["protein_name"].replace(',', '') + (' [' + amyloid["uniprot_start"] + '-' + amyloid["uniprot_end"] + ']' if amyloid["uniprot_start"] != "" else "")+ ',')
        lines.append(amyloid["sequence"] + ',')
        lines.append(str(amyloid["prion_domain"]) + ',')
        # positive amyloid classification
        lines.append('1' + '\n')
    json_file.close()

# Append to CSV file
with open(positiveDataset, 'a') as file:
    file.writelines(lines)
    file.close()


# ### Read CSVs into a Panda dataframe to generate training, testing & validation datasets
# For performance reasons, header names are replaced with indexes that reference an in-memory dictionary. A constant random seed may be assigned to ensure the same data is split across training, testing & validation.

# In[3]:


import pandas as pd
import random
from sklearn.model_selection import train_test_split, KFold

train_validation_ratio = 0.7
train_test_ratio = 0.7
df_train, df_test = pd.DataFrame(), pd.DataFrame()

def generateDataFrames(randomSeed):
    # Read raw data & combine 
    df_positive = pd.read_csv(positiveDataset).applymap(lambda x: x.strip() if type(x)==str else x)
    df_negative = pd.read_csv(negativeDataset).applymap(lambda x: x.strip() if type(x)==str else x)
    df_full = pd.concat([df_positive, df_negative], ignore_index=True)

    # Generate in-memory index for headers & compress dataframe
    datasetIndex = {}
    for index in range(len(df_full.index)):
        datasetIndex[index] = df_full.at[index, 'header']
        df_full.at[index, 'header'] = index

    # Split amyloid class
    df_notAmyloid = df_full[df_full['amyloid'] == 0]
    df_amyloid = df_full[df_full['amyloid'] == 1]

    # Train-validation split
    df_amyloid_trainSuperset, df_amyloid_validation = train_test_split(df_amyloid, train_size = train_validation_ratio, random_state = randomSeed)
    df_notAmyloid_trainSuperset, df_notAmyloid_validation  = train_test_split(df_notAmyloid, train_size = train_validation_ratio, random_state = randomSeed)

    # Train-test split
    df_amyloid_train, df_amyloid_test = train_test_split(df_amyloid_trainSuperset, train_size = train_test_ratio, random_state = randomSeed)
    df_notAmyloid_train, df_notAmyloid_test = train_test_split(df_notAmyloid_trainSuperset, train_size = train_test_ratio, random_state = randomSeed)

    # Concatenate splits of different labels into training, testing & validation sets
    df_train = pd.concat([df_amyloid_train, df_notAmyloid_train], ignore_index=True, sort=False)
    df_test = pd.concat([df_amyloid_test, df_notAmyloid_test], ignore_index=True, sort=False)
    df_valid = pd.concat([df_amyloid_validation, df_notAmyloid_validation], ignore_index=True, sort=False)
    df_full = pd.concat([df_train, df_test], ignore_index=True, sort=False)

    # Write preprocessed data
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    df_valid.to_csv('valid.csv', index=False)
    df_full.to_csv('full.csv', index=False)

    return datasetIndex


# ## Implement the model
# ### Import necessary libraries

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# set processing device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# data 

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torch.utils.data import ConcatDataset

# model

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# training

import torch.optim as optim
from tensorboardX import SummaryWriter
tensorboard_writer = SummaryWriter('./tensorboard_logs')

# evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns

# allegro trains integration
    # automation modules for hyperparameter optimization
from trains.automation import UniformParameterRange, UniformIntegerParameterRange
from trains.automation import HyperParameterOptimizer
from trains.automation.optuna import OptimizerOptuna


# ### Load data
# 

# In[5]:


# Tokenize peptide sequences by splitting into individual amino acids
def split(sequence):
    return [char for char in sequence] 

def declareFields():
    header = Field(sequential=False, dtype=torch.int, use_vocab=False, include_lengths=False)
    sequence = Field(tokenize=split, sequential=True, include_lengths=True, batch_first=True)
    amyloidLabel = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, include_lengths=False)
    prionLabel = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, include_lengths=False)

    fields = [('header', header), ('sequence', sequence), ('prion', prionLabel), ('amyloid', amyloidLabel)]
    # Create TabularDatasets for vocab & fold splits
    train, test = TabularDataset.splits(path="./", train="train.csv", test="test.csv", format='CSV', fields=fields, skip_header=True)
    full = ConcatDataset([train, test])

    # Vocabulary
    sequence.build_vocab(train)
    print("Vocabulary: " + str(sequence.vocab.stoi.items()))

    return full, fields, sequence

randomSeed = int(abs(np.random.normal() * 10))
datasetIndex = generateDataFrames(randomSeed)
fullDataset, fields, sequence = declareFields()
  


# In[ ]:


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


# ### Model
# A stacked, bidirectional gated recurrent unit (GRU) network will be applied to learn amyloidogenic & prionogenic relations in peptide sequences. 
# 
# This network is composed of an embedding layer to vectorize input sequences, two GRU layers, and a fully-connected linear output layer gated by sigmoidal activation.
# 
# Hyperparameters:
# - embedding & hidden layer dimensions
# - depth of LSTM 
# - LSTM bidirectionality
# - neuron dropout probability

# In[6]:


class GRU(nn.Module):

    def __init__(self, vocab, dimension=128, sequenceDepth = 64, dropoutWithinLayers = 0.3, dropoutOutput = 0.3):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), sequenceDepth)
        self.dimension = dimension
        self.GRU = nn.GRU(input_size=sequenceDepth,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropoutWithinLayers)
        self.dropOutput = nn.Dropout(p=dropoutOutput)

        # output layer
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.GRU(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.dropOutput(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# #### Second model
# A similar, but less performant, long-short term memory model is supplied here.

# In[7]:


class LSTM(nn.Module):

    def __init__(self, vocab, dimension=64):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 32)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=dimension,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.dropOnOutput = nn.Dropout(p=0.5)

        # output layer
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.dropOnOutput(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# ## Train

# ### Training function
# 
# 

# In[8]:


#### CSV row index sampler
def sampleCSV(csvFilePath, sampleType, sampler):
    df_full = pd.read_csv(csvFilePath)
    df_sample = df_full.iloc[sampler]
    df_sample.to_csv(destination_folder + '/sample' + sampleType + '.csv', index=False)
    return 'sample' + sampleType + '.csv'


# In[9]:


def train(vocab,
          lr,
          dimension,
          sequenceDepth,
          dropoutWithinLayers,
          dropoutOutput,
          batchSize,
          criterion,
          dataset,
          fields,
          foldCount,
          config,
          file_path,
          num_epochs = 5,
          amyloid = True,
          best_valid_loss = float("inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = {i:[] for i in range(foldCount)}
    valid_loss_list = {i:[] for i in range(foldCount)}
    global_steps_list = {i:[] for i in range(foldCount)}

    # K-fold cross-validation loop

    kfold = KFold(n_splits=foldCount, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        # sample elements in list of IDs
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainSamplePath = sampleCSV('full.csv', 'Train' + str(fold), train_ids)
        testSamplePath = sampleCSV('full.csv', 'Test' + str(fold), test_ids)

        # create datasets for current fold
         # Create TabularDatasets
        trainData, testData = TabularDataset.splits(path="./", train="output/sampleTrain" + str(fold) + ".csv", test="output/sampleTest" + str(fold) + ".csv", format='CSV', fields=fields, skip_header=True)

        # create iterators for current fold
            # sort by sequence length to keep batches consistent 
        train_iter =  BucketIterator(trainData, batch_size=batchSize, sort_key=lambda x: len(x.sequence),
                            device=device, sort=True, sort_within_batch=True)
        test_iter =  BucketIterator(testData, batch_size=batchSize, sort_key=lambda x: len(x.sequence),
                            device=device, sort=True, sort_within_batch=True)
        eval_every=len(train_iter) // 2

        # initialize model
        model = GRU(vocab=vocab, dimension=dimension, sequenceDepth=sequenceDepth, dropoutWithinLayers=dropoutWithinLayers, dropoutOutput=dropoutOutput).to(device)

        # load optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)


        # training loop
        model.train()
        for epoch in range(num_epochs):
            for (header, (sequence, sequence_len), prionLabel, amyloidLabel), _ in train_iter:           
                prionLabel = prionLabel.to(device)
                amyloidLabel = amyloidLabel.to(device)
                sequence = sequence.to(device)
                sequence_len = sequence_len.to(device)
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
                    train_loss_list[fold].append(average_train_loss)
                    valid_loss_list[fold].append(average_valid_loss)
                    global_steps_list[fold].append(global_step)

                    tensorboard_writer.add_scalar("total training loss", average_train_loss, global_step)
                    tensorboard_writer.add_scalar("total testing loss", average_valid_loss, global_step)

                    # resetting running values
                    valid_running_loss = 0.0
                    running_loss = 0.0
                    model.train()

                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                            .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter)*(foldCount),
                                    average_train_loss, average_valid_loss))
                    
                    torch.save(model, destination_folder + f'/model{fold}.pt')
                    save_metrics(file_path + f'/metrics{fold}.pt', train_loss_list[fold], valid_loss_list[fold], global_steps_list[fold])


    print('Finished Training!')
    return train_loss_list, valid_loss_list, global_steps_list


# ### Entry point for training amyloid-only model
# 
# Hyperparameters:
#  - Learning rate
#  - Epoch count
#  - Loss function
#  - Hidden layer size
#  - Hidden layer depth

# In[10]:


# hyperparameters
config = {'number_of_epochs': 6, 'batch_size': 16, 'dropout_within_layers': 0.3, 'dropout_output': 0.3, 'learning_rate': 0.0004, 'sequence_feature_depth':64, 'dimension':128}

# allegro initialization
from trains import Task
task = Task.init(project_name='agGRU', 
                 task_name='agGRU-Check Training')
allegroConfig_dict = task.connect(config)

training_losses, validation_losses, global_steps = train(
    vocab=sequence.vocab, lr=0.0004, criterion=nn.SmoothL1Loss(), fields=fields, dataset=fullDataset, foldCount=10, config=config, file_path = destination_folder, num_epochs=config['number_of_epochs'], dimension=config['dimension'], batchSize=config['batch_size'], sequenceDepth=config['sequence_feature_depth'], dropoutWithinLayers = config['dropout_within_layers'], dropoutOutput = config['dropout_output'])


# In[ ]:


#### optimize hyperparameters
from trains import Task
task = Task.create(project_name='agGRU', 
                 task_name='agGRU-Check Hyperparameter optimization')

optimizer = HyperParameterOptimizer(
    base_task_id='b7f9f1701c7c4b38b09ab68391e03c23',  
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=100, step_size=2),
        UniformIntegerParameterRange('dimension', min_value=16, max_value=800, step_size=2),
        UniformIntegerParameterRange('sequence feature depth', min_value=16, max_value=800, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=128, step_size=2),
        UniformParameterRange('dropout_output', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('dropout_between_layers', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('learning_rate', min_value=0.00001, max_value=0.01, step_size=0.000015),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='total_testing_loss',
    objective_metric_series='total_testing_loss',
    objective_metric_sign='min',  

    # setting optimizer 
    optimizer_class=OptimizerOptuna,
    
    # Configuring optimization parameters
    execution_queue='hyperparameter_optimization',  
    max_number_of_concurrent_tasks=2,  
    optimization_time_limit=60., 
    compute_time_limit=120, 
    total_max_jobs=20,  
    min_iteration_per_job=15000,  
    max_iteration_per_job=150000,  
)

optimizer.set_report_period(1) # setting the time gap between two consecutive reports
optimizer.start()
optimizer.wait() # wait until process is done
optimizer.stop() # make sure background optimization stopped


# ## Evaluate
# ### Training losses

# In[78]:


foldCount = 10
if len(global_steps[0]) < len(train_loss_list): global_steps[0].insert(0, 0)

# average losses 
totalTrainingLosses = pd.DataFrame(list(training_losses.values()))
totalTestingLosses = pd.DataFrame(list(validation_losses.values()))
totalTrainingLosses.to_csv(destination_folder + '/averageTrainingLosses.csv')
totalTestingLosses.to_csv(destination_folder + '/averageTestingLosses.csv')

averageTrainingLosses = totalTrainingLosses.mean(axis=0)
averageTrainingDeviation = totalTrainingLosses.std(axis=0)
averageTestingLosses= totalTestingLosses.mean(axis=0)
averageTestingDeviation = totalTestingLosses.std(axis=0)

# plot loss across folds
plt.figure(figsize=(36,8))
plt.title('Cross-validated loss (10x)')
for fold in range(foldCount):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + f'/metrics{fold}.pt')
    plt.plot(global_steps_list, train_loss_list, label=f'Train: {fold+1}')
    plt.plot(global_steps_list, valid_loss_list, label=f'Test: {fold+1}')
    plt.legend()
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.show()

# plot average loss across folds
plt.title('Averaged 10x cross-validation loss')
plt.plot(global_steps[0], averageTrainingLosses, label='Train')
plt.errorbar(global_steps[0], averageTestingLosses, averageTestingDeviation, label='Test', ecolor='black', elinewidth=0.5)
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show() 


# ### Confusion matrix, heuristics & external validation

# In[79]:


def kFoldEvaluate(model, foldCount, csvDatasetPath = None, threshold=.6):
    y_pred = []
    y_true = []

    def evaluate(dataPath, y_pred, y_true):
        testData = TabularDataset(path=dataPath, format="CSV", fields=fields, skip_header=True)
        test_iter =  BucketIterator(testData, batch_size=16, sort_key=lambda x: len(x.sequence),
                                device=device, sort=True, sort_within_batch=True)
        with torch.no_grad():
            for (header, (sequence, sequence_len), prionLabel, amyloidLabel), _ in test_iter:           
                prionLabel = prionLabel.to(device)
                amyloidLabel = amyloidLabel.to(device)
                sequence = sequence.to(device)
                sequence_len = sequence_len.to(device)
                output = model(sequence, sequence_len.cpu())

                output = (output > threshold).int()
                y_pred.extend(output.tolist())
                y_true.extend(amyloidLabel.tolist())

    model.eval()
    if not csvDatasetPath:
        for fold in range(foldCount):
            evaluate(destination_folder + "/sampleTest" + str(fold) + ".csv", y_pred, y_true)
    else:
        evaluate(csvDatasetPath, y_pred, y_true)
            
    return y_pred, y_true


# ### Classification report, confusion matrix & ROC

# In[80]:


def getHeuristics(y_true, y_pred, confusionMatrixTitle, ROCtitle, fold=None):
    cm = confusion_matrix(y_true, y_pred, labels=[1,0], normalize='true')
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues')

    ax.set_title(confusionMatrixTitle)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['no', 'yes'])
    ax.yaxis.set_ticklabels(['no', 'yes'])


    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label=f'Receiver operating characteristic (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(ROCtitle)
    plt.legend(loc="lower right")
    plt.show()

    print(f'Classification Report {fold + 1}:' if fold else 'Classification Report')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    return roc_auc

y_pred, y_true = dict(), dict()
bestModel = { 'model': None, 'fold': 0}
bestROC = 0
for fold in range(foldCount):
    model = torch.load(destination_folder + f'/model{fold}.pt')
    y_pred[fold], y_true[fold] = kFoldEvaluate(model, foldCount)
    roc = getHeuristics(y_pred[fold], y_true[fold], f"Confusion matrix: model {fold + 1} cross-validation", f"Receiver operating characteristic: model {fold + 1} cross-validation", fold)
    if roc > bestROC: 
        bestROC = roc
        bestModel['model'], bestModel['fold'] = model, fold + 1
print("bestROC: " + str(bestROC) + ', fold ' + str(bestModel['fold']))


# #### External validation

# In[81]:



validationDataPath = './valid.csv'

y_pred_external, y_true_external = dict(), dict()
bestExternalModel = { 'model': None, 'fold': 0}
bestExternalROC = 0
for fold in range(foldCount):
    model = torch.load(destination_folder + f'/model{fold}.pt')
    y_pred_external[fold], y_true_external[fold] = kFoldEvaluate(model, foldCount, validationDataPath)
    roc = getHeuristics(y_pred_external[fold], y_true_external[fold], f"Confusion matrix: model {fold + 1} external validation", f"Receiver operating characteristic: model {fold + 1} external validation", fold)
    if roc > bestExternalROC: 
        bestExternalROC = roc
        bestExternalModel['model'], bestExternalModel['fold'] = model, fold + 1
print("bestROC: " + str(bestExternalROC) + ', model ' + str(bestExternalModel['fold']))
print("best external model = best cross-validated model? " + ("True" if bestModel['fold'] == bestExternalModel['fold'] else "False"))


# 
# ## Apply
# Use the model to search for amyloidogenic sequences in the Gut Phage Database, and analyze results.

# In[33]:


def compressHeaders(FASTApath, CSVpath):
    print("Converting to CSV...")
    fastaToCSV(FASTApath, CSVpath, False)

    print("Reading CSV as dataframe...")
    df_realData = pd.read_csv(CSVpath, engine='python')

    # Generate in-memory index for headers & compress dataframe
    realDatasetIndex = {}
    print("Generating header index...")
    for index in range(len(df_realData.index)):
        realDatasetIndex[index] = df_realData.at[index, 'header']
        df_realData.at[index, 'header'] = index
    print("Writing compressed CSV...")
    df_realData.to_csv(CSVpath, index=False)
    
    return realDatasetIndex

def loadTabularData(CSVpath):
    # Load iterator
    realData = TabularDataset(path=CSVpath, format="CSV", fields=fields, skip_header=True)
    return realData

def loadIterator(tabularData):
    header = Field(sequential=False, dtype=torch.int, use_vocab=False, include_lengths=False)
    sequence = Field(tokenize=split, sequential=True, include_lengths=True, batch_first=True)

    fields = [('header', header), ('sequence', sequence)]

    realData_iter =  BucketIterator(tabularData, batch_size=16, sort_key=lambda x: len(x.sequence),
                                device=device, sort=True, sort_within_batch=True)
    # Vocabulary
    sequence.build_vocab(tabularData)
    return realData_iter, sequence

def evaluateRealData(model, iter, threshold=0.5):
    y_pred = []

    model.eval()
    with torch.no_grad():
        for (header, (sequence, sequence_len), prionLabel, amyloidLabel), _ in iter:           
            sequence = sequence.to(device)
            sequence_len = sequence_len.to(device)
            output = model(sequence, sequence_len.cpu())

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
    return y_pred


# ### Gut phage database
# 

# In[112]:


#### Parse FASTA into CSV

gpdDatasetPath = './GPD_proteome.faa'
gpdCSVDatasetPath = './GPD_proteome.csv'

gpdDatasetIndex = compressHeaders(gpdDatasetPath, gpdCSVDatasetPath)
print("done!")


# In[146]:


#### Chunk CSV into bins
gpdChunkFolder = "./outputChunks"

print("Reading data...")
df_gpd = pd.read_csv(gpdCSVDatasetPath)
df_gpd.drop(columns =["prion", "amyloid"])

print("Chunking...")
n = 760000
chunks = [df_gpd[i:i+n] for i in range(0,df_gpd.shape[0],n)]

for i in range(0,df_gpd.shape[0],n):
    df_gpd[i:i+n].to_csv(gpdChunkFolder + f"/gpd{int(i/n)}.csv", index=False)
    print(f"Saved chunk {i}.")

print("done!")


# In[83]:


#### Load first third of CSV chunks into tabular datasets
gpdTabularDatasets = list()

for i in range(3):
    print(f"loading TabularDataset {i + 1}...")
    gpdTabularDatasets += [loadTabularData(gpdChunkFolder + f"/gpd{int(i)}.csv")]
print("done!")


# In[ ]:


#### Load next third of CSV chunks into tabular datasets
gpdTabularDatasets = list()

for i in range(3, 6):
    print(f"loading TabularDataset {i + 1}...")
    gpdTabularDatasets += [loadTabularData(gpdChunkFolder + f"/gpd{int(i)}.csv")]
print("done!")


# In[84]:


#### Load last third of CSV chunks into tabular datasets
for i in range(6, 10):
    print(f"loading TabularDataset {i + 1}...")
    gpdTabularDatasets += [loadTabularData(gpdChunkFolder + f"/gpd{int(i)}.csv")]
print("done!")


# In[85]:


### Create iterators

gpdIterators = list()
for i in range(10):
    print(f"loading iterator {i + 1}...")
    gpdIteratorAndSequence = loadIterator(gpdTabularDatasets[i])
    gpdIterators += [gpdIteratorAndSequence[0]]
sequence = gpdIteratorAndSequence[1]
print("done!")


# #### Get predictions

# In[86]:


best_model = torch.load('./0.89' + f'/model2.pt')
gpd_predictions = list()

for i in range(10): 
    print(f"evaluating chunk {i}")
    gpd_predictions += evaluateRealData(best_model, gpdIterators[i])


# In[95]:


### Parse results into dict
import csv

csv_columns = ['protein']
gpdPositivesCSV = list()
gpdPositivesSet = set()
print("expanding headers into dict with predictions...")
for index, prediction in enumerate(gpd_predictions):
    gpdPositive = dict()
    if prediction == 1:
        gpdPositive["protein"] = gpdDatasetIndex[index]
        gpdPositivesSet.add(gpdDatasetIndex[index])
        gpdPositivesCSV.append(gpdResult.copy())

print("writing dict to CSV...")
csv_file = "./output/gpdPositives.csv"

with open(csv_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in gpdPositivesCSV:
        writer.writerow(data)


# #### Analyze predictions

# In[109]:


#### Load GPD annotations to find results with functional annotations

df_gpdAnnotations = pd.read_csv("./GPD_proteome_orthology_assignment.txt", sep="\t", dtype=str)


# ## Notes
# 
# ### Dataset
# 
# ### Proteins
# - S100-A9: false negative
#     - amyloidogenic in presence of calcium/zinc due to metallic ion binding affinity
# 
# 
