"""#Artificial Neural Network

#Installing Theano
its and open sourse numerical computationslibrary , very efficient for fast numerical computations 
and is based on numpy syntax 
This library not only runs on your cpu but also on your gpu
Forward and backward propagation both requires high computational power which we get from the gpu as it 
requires parallel computations

#Installing Tensorflow
Open Sourse numerical computational library that runs very fast computations and can run on cpu n gpu

THEANO AND TENSORFLOW ARE MOSTLY USED FOR RESEARCH AND DEVELOPMENT PURPOSE IN DEEP LEARNING FIELD
THAT MEANS IF YOU WANT TO USE THESE 2 LIBRARIES FOR DEEP LEARNING YOU WOULD USE THEM TO BUILD A DEEP NEURAL
NETWORK FROM SCRATCH.this is many lines of code

#Installing Keras
This envelops the above 2 libraries i.e its based on theano n tensorflow 
Help to build very powerful deep neural networks in few lines of code"""
# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
"""As our independent varibles have categories we need to encode the categorical varibles
We need to encode the categorical data before splitting data into train and test sets
We can see there are 2 columns with the categorical varibles
a)country (germany,spain,france) b)gender (male,female) so these are only 2 variables to encode"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
"""Our categorical varibles are not ordinal i,e there is no relational order between the categories
of the categorical variables i.e France isnt higher than Germany and Germany isnt higher than Spain etc
So we need to create dummy variables and we will do it only for country column as we remove 1 column
to avoid the DUMMY VARIABLE TRAP it wont be of any use in the gender column ."""
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]#this is done to remove the 1st column of X as to avoid the dummy variable trap


"""We need to apply feature scaling to the ANN as there high computations to be performed
and parallel computations also.
Also done to avoid dominance of independent variable over the other."""


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_1=StandardScaler()
X_train=sc_1.fit_transform(X_train)
X_test=sc_1.transform(X_test)

#Part 2 - Lets make the ANN

#Importing keras libraries and packages
#import keras #not happending ...giving error
#import tensorflow 
import keras

#Sequential module to initialise the neural network
from keras.models import Sequential

#Dense module required to build layers of our ANN
from keras.layers import Dense


#Initialise the ANN
#2 ways : 1)Defining the ANN layers 2)DEfining the graph
"""1st we create the object of the Sequential class and this object is nothing but the 
model itself i.e the neural network that will have a role of classifier here bcoz our 
problem is a classification problem where we have to predict a class"""
classifier = Sequential()

#Adding the input layer and first hidden layer
#Input layer will have 11 nodes as we have 11 independent variables
#we will choose recitifier activation fucntion for hidden layers 
#and sigmoid activation function for the output layer
classifier.add(Dense(output_dim=6 ,init='uniform',activation='relu',input_dim=11))
#output_dim is the number of nodes required in the hidden layer
#init means the initialisation of the weights nearby 0 with the uniform function
#activation means the activation function and relu is the rectifier function
#input_dim is the number of input i.e the independent variables =11 
#we need to specify the input_dim just for the 1st hidden layer as next hidden layers are not directly
#connected to the input layer 

#Adding a Second hidden layer
classifier.add(Dense(output_dim=6 ,init='uniform',activation='relu'))

#Adding the final layer i.e the output layer
#output_dim is the output varible n here its a binary output so we need only 1 
#if you are dealing with scenario where the ouput is a categorical variable then you need to change the output_dim
#and this will be equal to the no_of_o/p_variables and activation='soft_max'
#soft_max is a sigmoid fucntion but applied to a variable having more than 2 categories
#here we have 2 categories hence we use sigmoid function
classifier.add(Dense(output_dim=1 ,init='uniform',activation='sigmoid'))

#Comipling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#optimizer is the algorithm u want to use to adjust the weights to minimize the cost function
#adam is the schocastic gradient descent algo.
#sgd algo is based on the loss function which u need to optimize to find the optimal weights
#as we have used sigmoid in the ouput layer the loss funcn is the logarithmic function
#if we have 2 categorie(binary o/p) then this fucntion is called as binary_crossentropy
#and if we have more than 2 outcomes then its called categorical_crossentropy
#metrics argument is the criterion u choose to evaluate your model and typically we use the 
#accuracy criterion 
#so basically what happens is that when the weights are updated after each observation or after
#each batch of many obs the algorithm uses this accuracy criterion to impore the models 
#performance 

#Fitting our ANN to training set
#No_of_epochs i.e the number of times we are training our ANN on the whole trainig set
#accuracy is increased at each round i.e at each epoch
classifier.fit(X_train , y_train , batch_size=10 , nb_epoch=100)
#batch_size is the number of obs after which u want to update the weights
#epoch=> a round when the whole training set is passed through the ANN
#nb_epoch has been changed to epochs

#Part 3 : Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#this returns the probabilties if or not the person will leave the bank
y_pred = (y_pred > 0.5) #0.5 is the threshold
#this will give true and false results of the y_pred
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
#but for the confusion matrix we need the real results i.e yes/no 1/0 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy
 (1537+189)/2000
#this accuracy is on the test set on which we didnt train our model 