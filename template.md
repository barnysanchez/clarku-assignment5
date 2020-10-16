# Instructions for Assignment 4 submission 

- Please copy the contents of this markdown as the starting template of your assingment. 
- The resulting file from this programming assingment must be submitted to the private repo you submitted for my review during assignment one. 
- The name of the file has to follow  naming convention:  *lastname*_week5.ipynb
- There are 4 sections of this assignments required to receive full credit. 
- Please see that the expected file is a Jupyter notebook which contains Python code and markdown, just like shown in class and in my recap video from week 2. 

# Section 1) Problem description (0%)
In this assignment you will get to practice with a very simple neural network implementation from scratch. 
This neural network, will have to learn what the important features are in the data to produce the output. 
In particular, this neural net will be given an input matrix of zeros and ones. The output to each sample will be a single one or zero. The output will be determined by the number in the first feature column of the data samples. 


# Section 2) This is your starting code for this assignment (20%)

(If you copy this code block and execute it it will run. However in this section the request for full credit is to change the inputs and outputs to a function or functions that fetches the inputsANDoutputs.csv file and produces the needed input and output arrays for you. Failure to implement a function to do this will drop your score for this section to 0%. If you cannot figure it out and only alter the code to insert the values manually to proceed to the other sections, will only earn you 5%)

The link for the training data is the following:  https://github.com/barnysanchez/clarku-assignment4/raw/main/inputANDoutputs.csv



    import numpy as np 
    import matplotlib.pyplot as plt 

    # input data  (student to implement fetch function to grab inputs/outputs file and transform accordingly)
    # The location to fecth data is:  https://github.com/barnysanchez/clarku-assignment4/raw/main/inputANDoutputs.csv

    inputs = np.array([[0, 1, 0],
                    [0, 1, 1],])

    # output data
    outputs = np.array([[0], [0]])

    # create NeuralNetwork class
    class NeuralNetwork:

        # intialize variables in class
        def __init__(self, inputs, outputs):
            self.inputs  = inputs
            self.outputs = outputs
            # initialize weights as .50 for simplicity
            self.weights = np.array([[.50], [.50], [.50]])
            self.error_history = []
            self.epoch_list = []

        #activation function ==> S(x) = 1/1+e^(-x)
        def sigmoid(self, x, deriv=False):
            if deriv == True:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))

        # data will flow through the neural network.
        def feed_forward(self):
            self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

        # going backwards through the network to update weights
        def backpropagation(self):
            self.error  = self.outputs - self.hidden
            delta = self.error * self.sigmoid(self.hidden, deriv=True)
            self.weights += np.dot(self.inputs.T, delta)

        # train the neural net for 5 iterations
        def train(self, epochs=5):
            for epoch in range(epochs):
                # flow forward and produce an output
                self.feed_forward()
                # go back though the network to make corrections based on the output
                self.backpropagation()    
                # keep track of the error history over each epoch
                self.error_history.append(np.average(np.abs(self.error)))
                self.epoch_list.append(epoch)

        # function to predict output on new and unseen input data                               
        def predict(self, new_input):
            prediction = self.sigmoid(np.dot(new_input, self.weights))
            return prediction

    # create neural network   
    NN = NeuralNetwork(inputs, outputs)
    # train neural network
    NN.train()

    # create two new examples to predict                                   
    example = np.array([[1, 1, 0]])
    example_2 = np.array([[0, 1, 1]])

    # print the predictions for both examples                                   
    print(NN.predict(example), ' - Correct: ', example[0][0])
    print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

    # plot the error over the entire training duration
    plt.figure(figsize=(15,5))
    plt.plot(NN.epoch_list, NN.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()



# Section 3) Determine the number of epochs needed to achieve 99% prediction accuracy. Manipulate the plot to clearly show this (35%)

(Once with the code running after your previous modifications, determine the number of epochs needed to achieve 99% prediction accuracy. I will look at your plot for this but  put a comment in your Jupyter cells and call out that number clearly, the number of epochs you determined were needed. The plot will give me confirmation of your statement.)

# Section 4) Change the prediction function to use a softmax activation function instead of sigmoid and determine the number of epochs needed to achieve 99% prediction accuracy with the new prediction function. Manipulate the plot to clearly show this (45%)

(This is a repeat of the previous section that is testing your understanding of code adaptation and of the implementation of a softmax activation function. Once completed you need to show the same as in the previous section, specifically determine the number of epochs needed to achieve 99% prediction accuracy. I will look at your plot for this but  put a comment in your Jupyter cells and call out that number clearly, the number of epochs you determined were needed. The plot will give me confirmation of your statement.)
