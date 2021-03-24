# data 

from typing import Sequence
from torchtext.legacy import data
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from model import GRU
from torch.utils.data import ConcatDataset


# evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns

# set processing device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
plaacFasta = 'positiveExamples/yeastPrion.fasta'
negativeFasta = 'negativeExamples/negative.fasta'
positiveDataset = 'positiveExamples/positiveDataset.csv'
negativeDataset = 'negativeExamples/negativeDataset.csv'
destination_folder = "./output"


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Tokenize peptide sequences by splitting into individual amino acids
def split(sequence):
    return [char for char in sequence] 

def declareFields(train = False):
    header = Field(sequential=False, dtype=torch.int, use_vocab=False, include_lengths=False)
    sequence = Field(tokenize=split, sequential=True, include_lengths=True, batch_first=True)
    fields = [('header', header), ('sequence', sequence)]
    if train:
        amyloidLabel = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, include_lengths=False)
        prionLabel = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float, include_lengths=False)
        fields = [('header', header), ('sequence', sequence), ('prion', prionLabel), ('amyloid', amyloidLabel)]

    return fields, sequence

def compressHeaders(FASTApath, CSVpath):
    print("Converting to CSV...")
    fastaToCSV(FASTApath, CSVpath)

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

# def runModel(model, csvDatasetPath = None, threshold=.45):
#     y_pred = []

#     def evaluate(dataPath, y_pred):
#         fields, sequence = declareFields()
#         print(f"building iterator for {dataPath}... ")
#         testData = TabularDataset(path=dataPath, format="CSV", fields=fields, skip_header=True)
#         sequence.build_vocab(testData)
#         test_iter =  BucketIterator(testData, batch_size=16, sort_key=lambda x: len(x.sequence),
#                                 device=device, sort=True, sort_within_batch=True)
#         with torch.no_grad():
#             for (header, (sequence, sequence_len)), _ in test_iter:           
#                 sequence = sequence.to(device)
#                 sequence_len = sequence_len.to(device)
#                 output = model(sequence, sequence_len.cpu())

#                 output = (output > threshold).int()
#                 y_pred.extend(output.tolist())

#     model.eval()
#     evaluate(csvDatasetPath, y_pred)
            
#     return y_pred

def runModel(model, foldCount = 0, csvDatasetPath = None, csvFoldPathBase = None, threshold=.45):
    y_pred = []
    y_true = []

    def evaluate(dataPath, y_pred, y_true):
        fields, sequence = declareFields()
        testData = TabularDataset(path=dataPath, format="CSV", fields=fields, skip_header=True)
        sequence.build_vocab(testData)
        test_iter =  BucketIterator(testData, batch_size=16, sort_key=lambda x: len(x.sequence),
                                device=device, sort=True, sort_within_batch=True)
        with torch.no_grad():
            for (header, (sequence, sequence_len)), _ in test_iter:           
                sequence = sequence.to(device)
                sequence_len = sequence_len.to(device)
                output = model(sequence, sequence_len.cpu())

                output = (output > threshold).int()
                y_pred.extend(output.tolist())

    model.eval()
    if csvFoldPathBase:
        for fold in range(foldCount):
            evaluate(csvFoldPathBase + str(fold) + ".csv", y_pred, y_true)
    else:
        evaluate(csvDatasetPath, y_pred, y_true)
            
    return y_pred, y_true

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

### main method

#### Parse GPD FASTA into CSV
# gpdDatasetPath = './GPD_proteome.faa'
# gpdCSVDatasetPath = './GPD_proteome.csv'

# gpdDatasetIndex = compressHeaders(gpdDatasetPath, gpdCSVDatasetPath)
# print("done!")

# # ### Chunk CSV into bins
# # gpdChunkFolder = "./outputChunks"
# # print("Reading data...")
# # df_gpd = pd.read_csv(gpdCSVDatasetPath)
# # print("Chunking...")
# # n = 760000
# # chunks = [df_gpd[i:i+n] for i in range(0,df_gpd.shape[0],n)]
# # for i in range(0,df_gpd.shape[0],n):
# #     df_gpd[i:i+n].to_csv(gpdChunkFolder + f"/gpd{int(i/n)}.csv", index=False)
# #     print(f"Saved chunk {int(i/n)}.")
# # print("done!")

# fields, sequence = declareFields()

# ### Get predictions
# model = torch.load('./bestModel' + f'/publishedModel.pt')
# gpd_predictions = list()

# for i in range(10): 
#     print(f"evaluating chunk {i}")
#     chunk = runModel(model = model, csvDatasetPath='./outputChunks' + f"/gpd{i}.csv")
#     gpd_predictions += chunk
# print("done!")


# best_model = torch.load('./bestModel' + f'/publishedModel.pt')
# gpd_predictions = list()

# for i in range(10): 
#     print(f"evaluating chunk {i}")
#     predictionsChunk, null = runModel(best_model, foldCount=10, csvFoldPathBase='./outputChunks/gpd')
#     gpd_predictions += predictionsChunk
    
# print("done!")

# negativeDataset = './negativeExamples/negative.fasta'
# negativeCSVDataset = './negativeExamples/parsedNegativeDataset.csv'

# gpdDatasetIndex = compressHeaders(negativeDataset, negativeCSVDataset)
# predictionsChunk, null = runModel(best_model, foldCount=10, csvDatasetPath=negativeCSVDataset)

# print("done!")


# ### Write positives
# from Bio.SeqIO import FastaIO
# from Bio import SeqIO
# output_file = "./output/gpdPositivesAlt.fasta"
# fasta_out = FastaIO.FastaWriter(output_file, wrap=None)

# print("generating sequence-header index...")

# gpdProteinDict = SeqIO.index(gpdDatasetPath, "fasta")

# print("finding positives...")
# positiveCount = 0
# for index, prediction in enumerate(gpd_predictions):
#     if prediction == 1:
#         fasta_out.write_record(gpdProteinDict[gpdDatasetIndex[index]])
#         positiveCount += 1

# print(f"found {positiveCount} positives")
# print("done!")