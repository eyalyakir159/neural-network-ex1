import api
import numpy as np
import json
import matplotlib.pyplot as plt
import math
import time
class newwork:
    def __init__(self,sizes) -> None:
        #sizes is an array, each index says the amount of neurons in the layer 
        # [3,2,3] will have 3 layers (one hidden) with 3 2 and 3 neurons 
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        #randn returns a vectore of size y*1 , so each layer will have a vector that represents all the baiss for the layer
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #element in index i represent the connection betwen layer i and i+1. the row in the vector is the neruon number int he reciving layer and the coloumn is the sender
        #for example index one is the connection betwen the first layer and the second layer (first hidden layer) 
        # and the vector at location (0,2) is the weight betwen neruon 3 in the hidden layer to the first in the first layer

    def drop_out_function(self,X,drop_prob):
        # Apply dropout to input by zeroing out some of the elements with probability drop_prob
        # X: input tensor, drop_prob: dropout probability
        keep_prob = 1 - drop_prob
        mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return mask * X

    def feed_fowords(self,a,dropout=False,drop_prob=0):
        #given an imagae (input) feed it to the system
        #we need to run trought every layer and pass it to the next one etc...

        for b, w in zip(self.biases, self.weights):
            if dropout: #if need to drop out and im not at the result vector
                a=self.drop_out_function(a,drop_prob) #ignore some results
            a = sigmoid(np.dot(w, a)+b) #np.dot(w,a)+b is Z, so sigmoiod of z is the value.
        return a             
    

    def loss_mse(self,output,target):
        #mse is least square cost function
        mse = np.sum((output-target)**2)/(len(output)*2)
        return mse
    def loss_log(self,output,target):
        #wierd formula
        s1 = target*np.log(output)+(1-target)*np.log(1-output)
        return -np.sum(s1)
    def backprop(self,x,y):
        """
        idea behind the backprop for single input vector
        1) we need to feed it into the system and collect the z's and activiation 
        2) after reciving final resust via the result layer we need to caculate the error
        3) get the delta (the error times the simgod prime function)
        4) after reciving the delta we need to caculat the error backwards, we do this by muliplaying by the transpose weights
        intoation - multiplcation with transpone weight go back in time and normal weights go fowords in time
        5) after getting all the errors send them back
        """
        #we need to keep track of z's and activiasion

        activisions = [x]
        z_values = []
        #step one, feed fowords
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x)+b
            z_values.append(z)
            x = sigmoid(z) #np.dot(w,a)+b is Z, so sigmoiod of z is the value.
            activisions.append(x)
        
        #step two , caculate error
        error = activisions[-1]-y
        #get delta
        delta = error*sigmoid(x,derivative=True)
        delta_b = delta
        delta_w = np.dot(delta,activisions[-2].transpose())

        """
           we need to do it backwards now and keep track of the errors and the delta, to do so we will inisilaize 2 arrays 
           with 0's and start thier last vectore to be error  and delta 
           each vector will be of size 
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(b.shape) for b in self.weights]

        nabla_b[-1],nabla_w[-1]=delta_b,delta_w
        #now we need to follow the same procses and do it all the way from layer n to index 0
        for l in range(2,self.num_layers):
            # we need to follow the proces, get z than get a than get error than delta than delta_b,w
            #step one, we got z,a before in the feed trough
            z = z_values[-l]
            sp = sigmoid(z,derivative=True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activisions[-l-1].transpose())
        return (nabla_b,nabla_w)
    def SGD(self,data_reader:api.data_reader,learning_rate,ephocs,branch_size,l1=False,l2=False,dropout=False,lamda=0,drop_prob=0):
        """
            idea behind the method
            1) data_reader has all the data we need. for more accuracy we need to split it into branchs and shufle the data.
            2) we need to run each sample for the epchos in each iteration we need to do the following:
                * feed the data sample and get the deravatives 
                * update the basis and weights by the following eqastion (aginst the gradient)
                w = w-grad(w)*learning_reate
                b = b-grad(B)*learning_rate

            note to myself - we run echos times on each branch.
        """
        data_reader.get_batchs(branch_size)
        time_array = []
        max_grad_array = []
        min_grad_array = []
        mean_grab_array = []
        log_loss_array=[]
        accrucly_array = [] 
        for j in range(ephocs):
            data_reader.shuffle_train()
            start = time.time()
            min_grad = math.inf
            max_grad = 0     
            log_loss = 0 
            for mini_batch in data_reader.batches:
                mean_grad_b,mean_grad_W = self.update_mini_branch(mini_batch, learning_rate,l1,l2,lamda,data_reader.train_data_len)
                for data in mini_batch:
                    log_loss+=self.loss_log(self.feed_fowords(data.data_to_vector(),dropout,drop_prob),data.target_to_vector())
                gradient = []
                for b in mean_grad_b:
                    gradient.extend(list(b.flatten()))
                for w in mean_grad_W:
                    gradient.extend(list(w.flatten()))
                norm = np.linalg.norm(gradient)
                min_grad , max_grad = min(norm,min_grad),max(norm,max_grad)
            accrucly_array.append(self.evalouate(data_reader))
            log_loss_array.append(log_loss)
            min_grad_array.append(min_grad)
            max_grad_array.append(max_grad)
            mean_grab_array.append(np.linalg.norm(gradient))
            end = time.time()
            print ("Epoch {0} complete".format(j))
            time_array.append(end-start)

        self.save("branch_size_"+str(branch_size)+"_learning_rate_"+str(learning_rate)+"_epchos_"+str(ephocs)+"_network_"+str(self.sizes)+"drop_"+str(dropout)+"_l1_"+str(l1)+"_l2_"+str(l2))
        return ephocs,log_loss_array,time_array,mean_grab_array,min_grad_array,max_grad_array,accrucly_array

    def update_mini_branch(self,branch,learning_rate,l1,l2,lamda,n):
        #we need to sum every data unit contrubusion to the gradient and take his mean
        mean_grab_b = [np.zeros(b.shape) for b in self.biases]
        mean_grad_W = [np.zeros(w.shape) for w in self.weights]
        for data in branch:
            delta_nabla_b, delta_nabla_w = self.backprop(data.data_to_vector(),data.target_to_vector()) #get the grad for every data unit
            mean_grab_b = [nb+dnb for nb, dnb in zip(mean_grab_b, delta_nabla_b)] #sum it all and mean it
            mean_grad_W = [nw+dnw for nw, dnw in zip(mean_grad_W, delta_nabla_w)] #^^
        if l1:
            self.weights = [w-(learning_rate/len(branch))*nw - learning_rate*np.sign(w)*(lamda/n)
                            for w, nw in zip(self.weights, mean_grad_W)] #go to the opisite side of the gradiet in the weights
        elif l2:
            self.weights = [(1-learning_rate*(lamda/n))*w-(learning_rate/len(branch))*nw
                for w, nw in zip(self.weights, mean_grad_W)]
        else:
            self.weights = [w-(learning_rate/len(branch))*nw
                            for w, nw in zip(self.weights, mean_grad_W)] #go to the opisite side of the gradiet in the weights
        self.biases = [b-(learning_rate/len(branch))*nb 
                       for b, nb in zip(self.biases, mean_grab_b)] #^^^
        return mean_grab_b,mean_grad_W
        #self.weights-mean_grad_W*learning_rate/len(branch) need to work as vectors
        #self.biases = self.biases-mean_grad_b*learning_rate/len(branch)

    def evalouate(self,data_reader):
        """given test data we need to see how many were predicet correcly"""
        counter = 0
        for data_unit in data_reader.test_data:
            output = self.feed_fowords(data_unit.data_to_vector())
            output_number_represantion = np.argmax(output)
            if data_unit.target == output_number_represantion:
                counter+=1
            data_unit.view_data()
            print(data_unit.target,output_number_represantion)
        return counter/len(data_reader.test_data)
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = newwork(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def sigmoid(z,derivative =False):
    """The sigmoid function."""
    if derivative:
        return sigmoid(z)*(1-sigmoid(z))
    return 1.0/(1.0+np.exp(-z))




bsize = [5,20,50,100]
network_sizes = load('branch_size_10_learning_rate_1_epchos_30_network_[784, 20, 10]drop_False_l1_False_l2_False')

b = api.data_reader(60000,9000,1000)
network_sizes.evalouate(b)

if False:
    plt.figure(1)
    plt.plot(list(range(ephocs)), log_loss_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('log-loss')
    plt.figure(2)
    plt.plot(list(range(ephocs)), time_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('training time per epcho')
    plt.figure(3)
    plt.plot(list(range(ephocs)), mean_grab_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('mean grad')
    plt.figure(4)
    plt.plot(list(range(ephocs)), min_grad_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('minimum of the grad')
    plt.figure(5)
    plt.plot(list(range(ephocs)), max_grad_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('maximum of the grad')
    plt.figure(6)
    plt.plot(list(range(ephocs)),accrucly_array)
    plt.xlabel('epchos')
    # naming the y axis
    plt.ylabel('acrruacly presantage')
    plt.show()


