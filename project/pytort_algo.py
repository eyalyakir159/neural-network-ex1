import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import json 
import torch.nn.functional as F


input_size = 784 # 28x28 image dat
hidden_size = 512 # hideen layer
output_size = 10 #numbers 0 to 9
num_epochs = 50
batch_size = 100
drop_rate = 0.3
learing_rate = 0.01
weight_decay = 1e-5 
add_fake = True
# Set up dataset, and dataloader





def change_data(add_fake):
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    transform_rotate = transforms.Compose([    transforms.RandomRotation(degrees=(-30, 30)),    transforms.ToTensor()]) #roate image , to tensor
    transform_noise = transforms.Compose([ #add noise to image
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0., 1.)
    ])
    train_data_with_noise = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transform_noise, 
                                            download=True) #get the data from facebook data base 
    train_data_with_roation = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transform_rotate, 
                                            download=True) #get the data from facebook data base 
    train_data_without_noise=torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        transform=transforms.ToTensor(), 
                                        download=True) #get the data from facebook data base 
    if add_fake:
        return torch.utils.data.ConcatDataset([train_data_without_noise, train_data_with_noise,train_data_with_roation])
    else:
        return train_data_without_noise

train_dataset = change_data(add_fake=add_fake)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         transform=transforms.ToTensor()) #get the data from facebook data base and turn in into tensor object (like we used to turn it into numpy)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True) #unpack it and shuffle
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False) #unpack it no need to shufe becues its test data
print("amount of data",len(train_dataset))
def evaluate(model):
# Test the model
    model.eval() #tels the algo that currect lines wont train him but just use him
    with torch.no_grad(): #make sure what we do here wont be saved and updated later to the weights - not training just using the system
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader: #run on all data test
            #images = images.reshape(-1, 28*28) #turn branch of images into long vector so the system will know what to do with it
            labels = labels # get the image target
            outputs = model(images) #get what the system result
            # max returns (value ,index)
            #_, predicted = torch.max(outputs.data, 1)  #get arg max of resulst aka the number the system think it is
            _, predicted = torch.max(outputs, 1)  #get arg max of resulst aka the number the system think it is

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item() #check how many images in the branch were prediceted

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')
    model.train()
    return acc


    
    

# Define a model using PyTorch
class MisfitModel(torch.nn.Module):
 def __init__(self, input_size, hidden_size, output_size):
   super().__init__()
   self.l1 = nn.Linear(input_size, hidden_size) #fisrt layer
   self.l2 = nn.Tanh() #second layer, we use simgod but we can chagne to tanh or relu or w.e was in the book
   self.l3 = nn.Linear(hidden_size, output_size)  #lst layer
   self.dropout = torch.nn.Dropout(drop_rate) #applay drop out with dropout rate
  
 def forward(self, x):
   x = self.dropout(torch.relu(self.l1(x))) #go trought first layer and applay dropout
   x = self.dropout(torch.relu(self.l2(x))) #go trought second layer and applay dropout
   x = self.l3(x) #get the result do not applay drop out so we wont lose the information
   return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5) #input is  an array of images with one dimintion (grey scale) with images of 28*28, so we will turn it into 20 sub filters with siez of 24*24
        self.pool = nn.MaxPool2d(2, 2) #the data and max it , we will have 20 arrays with images of size 12*12
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5) #make another 20 images so we will have 20 arrays of ize 8*8 than turn it into 20 arrays of 4*4 
        self.fc1 = nn.Linear(20*4*4, 40) #so the input will be 20*4*4.
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10) #out put has 10 options so 10 outputs
        self.dropout1 = nn.Dropout(p=drop_rate)  
        self.dropout2 = nn.Dropout(p=drop_rate)  
        print(drop_rate)

    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  #max filter and first conv layer
        x = self.pool(F.relu(self.conv2(x))) #max filter and second conv layer
        x = x.view(-1, 20 * 4 * 4)     #turn it into long vector to feed theh liner system
        x = F.relu(self.fc1(x))    
        x = self.dropout1(x)  #apply drop out
        x = F.relu(self.fc2(x)) 
        x = self.dropout2(x)   #apply drop out
        x = self.fc3(x)        
        return x

def make_model(l2,weight_decay=0):
    # Set up the model, dataset, and dataloader
    #model = MisfitModel(input_size, hidden_size, output_size) #build thhe model
    model = ConvNet()
    # Set up the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss() #cross loss function, we can do mse and moore functions
    if l2:
        optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate, weight_decay=weight_decay) #gradient decent, weight decay = l2 regolation
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate) #gradient decent, weight decay = l2 regolation

    # Train the model
    acc_per_echo = []
    for epoch in range(num_epochs): #for each epcho
        acc_per_echo.append(evaluate(model=model))
        for i, (data, target) in enumerate(train_loader): #for each data branch
            # Forward pass
            #data = data.reshape(-1, 28*28) #turn in into vector so the system will know what to do with it
            output = model(data) #feed the system
            loss = criterion(output, target) #get the loss
            
            # Backward pass and optimization step
            optimizer.zero_grad() #face book algo has a map and saves all the data backwards, when we do stuff witht he sytem and we dont want it to effect the network we need to do no_grad or resat the grad
            loss.backward()
            optimizer.step() #do opimization step (update weights baisis etc)
        print (f'Epoch [{epoch+1}/{num_epochs}]')
    return model,acc_per_echo


# model_with_l2,acc_per_echo1 = make_model(l2=True,weight_decay=weight_decay)

