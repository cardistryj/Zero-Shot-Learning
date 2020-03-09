import numpy as np
from config import *
from dataset18 import Dataset

import kmean_methods as kmeanMethods

import torch
import sys
import copy

import torch.utils.data as Data

path_dataset='../data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'

dataset = DataSet(path_dataset)
X, lbl, snrs, mods = dataset.getX()
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
num_class = len(classes)

test_dataset = Data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
test_loader = Data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

def cal_acc_softmax():
    class_correct = torch.ones(len(classes)).to(device)
    class_total = torch.ones(len(classes)).to(device)
    
    for data in test_loader:
        images, labels = data
        print(labels.size())
        outputs = model(images.to(device))
        predicted = torch.max(outputs.data, 1)[1]
        labels = torch.max(labels.long(), 1)[1].to(device)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[int(label)] += c[i]
            class_total[int(label)] += 1

    aver_acc=0
    for i in range(len(classes)):
        acc=class_correct[i]/class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * acc))
        aver_acc+=acc
    print('Overall accuracy:{}'.format(aver_acc/len(classes)))


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('GPU is available...')
        device = torch.device("cuda")

    dataset.splitOneShotData()
    train_map,test_map,one_shot_map,untrain_test=dataset.getMapData()
   
    model.to(device)
    print('Loading model from {}'.format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print('With {} epochs training...'.format(checkpoint['epoch']))

    softmax_path=softmax_path.format(modification,featureDim,checkpoint['epoch'],date)
    with torch.no_grad():
        model.eval()

        print('Calculating softmax accuracy')
        cal_acc_softmax()

    print('end')
