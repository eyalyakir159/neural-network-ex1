
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import gzip
import pickle




class data_reader:
    def __init__(self,train_data_len,test_data_len,validation_data_len) -> None:
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.validation_data_len = validation_data_len
        self.train_data,self.test_data,self.validation_data = self.pull_data(self.train_data_len,self.test_data_len,self.validation_data_len)
        self.batches = []

    def pull_data(self,train_data_len,test_data_len,validation_data_len):
        data = datasets.load_digits()
        #get the data splited into gruopes
        train_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len)]
        validation_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len,train_data_len+validation_data_len)]
        test_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len+validation_data_len,train_data_len+validation_data_len+test_data_len)]
        return train_data,test_data,validation_data


    def shuffle_train(self):
        if self.batches:
            for j in self.batches:
                random.shuffle(j) 
            random.shuffle(self.batches)


    def get_batchs(self,size):
        if not self.batches:
            size = max(1, size)
            temp = [self.train_data[i:i+size] for i in range(0, len(self.train_data), size)]
        self.batches = temp
        return self.batches

    def update_batchs(self,batches):
        self.batches = batches
    

class data_unit:
    counter = 0 
    def __init__(self,data,target) -> None:
        data_unit.counter +=1
        self.data = data
        self.target = target
        self.data_unit_number = data_unit.counter
    def view_data(self):
        fig = plt.figure()
        plt.imshow(self.data,cmap = plt.cm.gray_r)
        plt.show()
    def __str__(self) -> str:
        return self.data_unit_number
    def data_to_vector(self):
        x = np.array(self.data).flatten()
        return x.reshape(len(x),1)
    def target_to_vector(self):
        a = np.zeros((10,1))
        a[self.target]=1
        return a

