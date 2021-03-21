# data 

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import ConcatDataset
import pandas as pd

def generateDataFrames(randomSeed, positiveDataset, negativeDataset, train_test_ratio, train_validation_ratio):
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
