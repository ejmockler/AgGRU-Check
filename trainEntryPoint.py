import numpy as np
import torch
import torch.nn as nn

from train import train
from loadData import generateDataFrames, declareFields
# set processing device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

plaacFasta = 'positiveExamples/yeastPrion.fasta'
negativeFasta = 'negativeExamples/negative.fasta'
positiveDataset = 'positiveExamples/positiveDataset.csv'
negativeDataset = 'negativeExamples/negativeDataset.csv'
destination_folder = "./output"

train_validation_ratio = 0.7
train_test_ratio = 0.7


randomSeed = int(abs(np.random.normal() * 10))
datasetIndex = generateDataFrames(randomSeed, positiveDataset, negativeDataset, train_test_ratio, train_validation_ratio)
fullDataset, fields, sequence = declareFields()

def run(allegroJob=False):
    # train entry point
    # hyperparameters
    config = {'number_of_epochs': 6, 'batch_size': 16, 'dropout_within_layers': 0.3, 'dropout_output': 0.3, 'learning_rate': 0.0004, 'sequence_feature_depth':64, 'dimension':128}
    if allegroJob:
        # allegro initialization
        from trains import Task
        task = Task.init(project_name='agGRU', 
                        task_name='agGRU-Check Training')
        allegroConfig_dict = task.connect(config)

    return train(
    vocab=sequence.vocab, lr=0.0004, criterion=nn.SmoothL1Loss(), fields=fields, dataset=fullDataset, foldCount=10, device=device, file_path = destination_folder, num_epochs=config['number_of_epochs'], dimension=config['dimension'], batchSize=config['batch_size'], sequenceDepth=config['sequence_feature_depth'], dropoutWithinLayers = config['dropout_within_layers'], dropoutOutput = config['dropout_output'])

run(allegroJob=False)