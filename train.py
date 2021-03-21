
# training

import torch.optim as optim
from model import GRU
from loadData import save_metrics
import torch
import pandas as pd
from tensorboardX import SummaryWriter
tensorboard_writer = SummaryWriter('./tensorboard_logs')

from torchtext.legacy.data import TabularDataset, BucketIterator
from sklearn.model_selection import KFold

#### CSV row index sampler
def sampleCSV(csvFilePath, sampleType, sampler, destination_folder = './output', ):
    df_full = pd.read_csv(csvFilePath)
    df_sample = df_full.iloc[sampler]
    df_sample.to_csv(destination_folder + '/sample' + sampleType + '.csv', index=False)
    return 'sample' + sampleType + '.csv'


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
          device,
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
                    
                    torch.save(model, file_path + f'/model{fold}.pt')
                    save_metrics(file_path + f'/metrics{fold}.pt', train_loss_list[fold], valid_loss_list[fold], global_steps_list[fold])


    print('Finished Training!')
    return train_loss_list, valid_loss_list, global_steps_list
