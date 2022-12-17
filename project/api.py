
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import gzip
import requests
import gzip
import _pickle as cPickle


class data_reader:
    def __init__(self,train_data_len,test_data_len,validation_data_len) -> None:
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.validation_data_len = validation_data_len
        self.train_data,self.test_data,self.validation_data = self.pull_data_2(self.train_data_len,self.test_data_len,self.validation_data_len)
        self.batches = []

    def pull_data(self,train_data_len,test_data_len,validation_data_len):
        data = datasets.load_digits()
        #get the data splited into gruopes
        train_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len)]
        validation_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len,train_data_len+validation_data_len)]
        test_data = [data_unit(data.images[i],data.target[i]) for i in range(train_data_len+validation_data_len,train_data_len+validation_data_len+test_data_len)]
        
        
        return train_data,test_data,validation_data

        
    def add_fake_data(data):
        pass
        """---ideas to add fake data---
            rotate image with fixed degree
            add noise aka gasuian distrabistion with 1% from max value
            blue the data with a filter"""


    def pull_data_2(self,train_data_len,test_data_len,validation_data_len):
        training_data = self.file_to_sampels('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',60000) #max 60000
        test_data = self.file_to_sampels('t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz',10000) #max 10000
        both = training_data+test_data
        try:
            return both[0:train_data_len],both[train_data_len+1:train_data_len+1+test_data_len],both[train_data_len+2+test_data_len:train_data_len+2+test_data_len+validation_data_len]
        except:
            print("big dims, we only have 80k images")
    def file_to_sampels(self,filename_data,filename_label,num_images):
        f = gzip.open(filename_data,'r')
        unit_data_in_array=[]
        image_size = 28
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size)
        f = gzip.open(filename_label,'r')
        f.read(8)
        for i in range(num_images):   
            buf = f.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            unit_data_in_array.append(data_unit(data[i]/256,label[0]))
        return unit_data_in_array

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


def dowlond_data():
    # The URL for the MNIST dataset
    mnist_url = "http://yann.lecun.com/exdb/mnist/"

    # The filenames for the train and test sets
    train_data_filename = "train-images-idx3-ubyte.gz"
    train_labels_filename = "train-labels-idx1-ubyte.gz"
    test_data_filename = "t10k-images-idx3-ubyte.gz"
    test_labels_filename = "t10k-labels-idx1-ubyte.gz"

    # Download the train data
    print("one")

    response = requests.get(mnist_url + train_data_filename)
    with open(train_data_filename, "wb") as f:
        f.write(response.content)
    print("2")
    # Download the train labels
    response = requests.get(mnist_url + train_labels_filename)
    with open(train_labels_filename, "wb") as f:
        f.write(response.content)
    print("3")

    # Download the test data
    response = requests.get(mnist_url + test_data_filename)
    with open(test_data_filename, "wb") as f:
        f.write(response.content)
    print("4")

    # Download the test labels
    response = requests.get(mnist_url + test_labels_filename)
    with open(test_labels_filename, "wb") as f:
        f.write(response.content)







