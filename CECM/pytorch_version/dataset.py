import numpy as np
import pickle
import random


class DataSet(object):
    def __init__(self, path, unit_num = 1000, rate = 0.8, num_seen_class=8, seed=2019):
        self.path = path
        self.unit_num = unit_num
        self.num_seen_class = num_seen_class
        self.seed = seed

        self.loadDataSet(path, self.unit_num)
        self.regularizeData()
        self.splitData(rate,num_seen_class, seed)

    # public methods
    def getX(self):
        return self.X, self.lbl, self.snrs, self.mods
    def getTrainAndTest(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.train_class
    def getTrainIndex(self):
        return self.train_idx
    def getTestIndex(self):
        return self.test_idx
    def getMapData(self):
        return self.train_map,self.test_map,self.one_shot_map,self.untrain_test_map

    def loadDataSet(self, path, unit_num):
        # 1. load dataset
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
        # print(Xd.keys())
        # print(len(set(map(lambda x: x[0], Xd.keys()))))

        # snrs(20) = -20 -> 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        print(self.mods)
        print(self.snrs)

        self.class_count=[0]
        count=0

        for mod in self.mods:
            for snr in self.snrs:
                if not snr in (18,16):
                    continue

                single_data=Xd[(mod, snr)][:]
                self.X.append(single_data)
                count+=single_data.shape[0]

                print('Shape:{}'.format(single_data.shape[0]))
                for i in range(single_data.shape[0]):
                    self.lbl.append((mod, snr))

            self.class_count.append(count)

        self.X = np.vstack(self.X)
        print(self.X.shape)


    def regularizeData(self):
        for i in range(2):
            for j in range(128):
                self.X[:,i,j]=(self.X[:,i,j]-np.min(self.X[:,i,j]))/(np.max(self.X[:,i,j])-np.min(self.X[:,i,j]))

#        print(np.max(self.X))


    def to_onehot(self, yy):
        yy1 = np.zeros([len(yy), len(self.mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    def splitOneShotData(self,rate=0.8):
        ###############
        untrain_class=list(set(range(0,len(self.mods)))-set(self.train_class))
        print(untrain_class)
        ###############

        self.one_shot_sample_idx=[]
        self.untrain_test_idx=[]
        self.one_shot_map={}
        self.untrain_test_map={}

        ##########
        for certain_class in untrain_class:
        ##########
        # for certain_class in range(0,len(self.mods)):
            class_idx=list(range(self.class_count[certain_class],self.class_count[certain_class+1]))
            one_shot_sample_class_idx=list(np.random.choice(class_idx,size=10,replace=False))
            untrain_test_class_idx=list(np.random.choice(list(set(class_idx)-set(one_shot_sample_class_idx)), size=int((self.class_count[certain_class+1]-self.class_count[certain_class])*(1-rate)+1), replace=False))
            self.one_shot_map[untrain_class.index(certain_class)+len(self.train_class)]=self.X[one_shot_sample_class_idx]
            self.untrain_test_map[untrain_class.index(certain_class)+len(self.train_class)]=self.X[untrain_test_class_idx]
            # self.one_shot_map[certain_class]=self.X[one_shot_sample_class_idx]
            # self.untrain_test_map[certain_class]=self.X[untrain_test_class_idx]
            self.untrain_test_idx+=untrain_test_class_idx
        print(len(self.untrain_test_idx))

    def splitData(self, rate=0.8, num_seen_class = 8, seed=2019):
        np.random.seed(seed)

        ############## used to select seen class
        #train_class=np.random.choice(range(0,len(self.mods)),size=num_seen_class,replace=False).tolist()+[8]
        
        train_class=list(set(range(11))-set([]))
        self.train_class=train_class

        print(train_class)
        ############### 

        # self.valid_idx=[]
        self.train_idx=[]
        self.test_idx=[]
        self.train_map={}
        self.test_map={}

        ################ as aforementioned
        for certain_class in train_class:
        ################
        #for certain_class in range(0,len(self.mods)):
            class_idx=list(range(self.class_count[certain_class],self.class_count[certain_class+1]))
            class_train_idx=list(np.random.choice(class_idx, size=int((self.class_count[certain_class+1]-self.class_count[certain_class])*rate), replace=False))
            class_test_idx=list(set(class_idx)-set(class_train_idx))
            self.train_map[train_class.index(certain_class)]=self.X[class_train_idx]
            # self.train_map[certain_class]=self.X[class_train_idx]
            
            self.test_map[train_class.index(certain_class)]=self.X[class_test_idx]
            # self.test_map[certain_class]=self.X[class_test_idx]

            self.train_idx+=class_train_idx
            self.test_idx+=class_test_idx

        random.shuffle(self.train_idx)
        random.shuffle(self.test_idx)
        self.X_train=self.X[self.train_idx]

        # self.X_train.resize([176000,256])      #to be removed

        # print(self.X_train.shape)

        self.X_test = self.X[self.test_idx]

        # self.X_test.resize([44000,256])        #to be removed

        self.Y_train = self.to_onehot(list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.train_idx)))
        self.Y_test = self.to_onehot(list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.test_idx)))

        # self.Y_train = self.to_onehot(list(map(lambda x: self.mods.index(self.lbl[x][0]), self.train_idx)))
        # self.Y_test = self.to_onehot(list(map(lambda x: self.mods.index(self.lbl[x][0]), self.test_idx)))

        classes = self.mods

        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.mods
