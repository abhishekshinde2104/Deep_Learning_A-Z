#Importing Librabries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset'
dataset=pd.read_csv("Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#columns ->attributes
#rows are the customers
"""So all the customers are the inputs of the neural networks
These input points are going to be mapped to a new output space and in between i/p and o/p space we this neural network
composed of neurons , each neuron being initialised as a vector of weights = same size as vector of customer = 
vector of 15 elements as 15 columns 
And for each customer id the o/p will be the neuron that is closet to customer = WINNING NODE 
For each customer the winning node is the most similar neuron for the customer 
And then we provide a NEIGHBOURD FUNCTION like guassian neighbourd func to update the weights of the winning node
to bring it closer 
And we do this for all customers in i/p space and we repeat that again and each time we repeat it the o/p space decreases
and loses dimensions and at last it reaches a point where the reduction stops """

######################################################################################

"""So while detecting the frauds , the frauds are actually the outlying neurons in the 2D som 
simply bcoz the outlying neurons are far from the neurons that follow the rules
How to detect outlying neurons ??? ==> MID mean interneuron distance 
That means in our som for each neuron we`re going to compute the mean of the eucledian distances between this neuron 
and the neurons in its neighbourhood and this neighbourdhood is defing manually 
We do this for all the neurons that we picked and find out the outliers and this will ultimately detect frauds 
And then we will use an inverse function to map this outlier to the input space to get the customer id  """

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

#Training the SOM
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
# x,y->size of the grid
#input_len->no of features in X
#sigma->radius of different neighbourhood 
#learning_rate-->decides by how much the weights are updated durting each iteration 
#decay_function-->used to increase convergence
#random_Seed-->

#now before training our model we need to initialise our weights

som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)  #treaining of model


#visualising the results
from pylab import bone,pcolor,colorbar,plot,show

bone()#initialisation of the window
#som.distance_map () ==> this mtd will return all the mean neuron distances in 1 matrix
pcolor(som.distance_map().T)
colorbar() #legend

#white colored boxes have large MIDs and therefore they are outliers and potential frauds

#markers
#red-cirlces==>customers who didnt get approval
#green-squares==>customers who got apporval

markers=['o','s'] #o->cirlce and s->square
colors=['r','g']#cirlce->red and square->green

#now we will loop over all the customers and then for each customer we will get the winning node and dependent on
#whether the customer got approval or not and color it red-cirlce if customer didnt get apoorval and 
#green-square if it got approval

for i,x in enumerate(X):
    w=som.winner(x)#get_winning_node 
    plot(w[0]+0.5 , w[1]+0.5 , markers[y[i]] , markeredgecolor=colors[y[i]],markerfacecolor='None',
         markersize=10,markeredgewidth=2)
    #w[0],w[1] are the coordinates of winning node and specifically lower left coordinates 
    """here y[i] is the vallue of dependent variable for that customer i.e 0->not approval and 1->yes approval
    if customer didnt get approval then y[i]=0 and for that a circle is drawn and when y[i]=1 then square is made
    $markerededgecolor will only color the edge of the marker
    but we wont color inside of the marker"""
    

#i->different values of indices of our customer database i.e rows =619
#x->vector of customers i.e all the columns for i

#in the colorbar 1 means fraud customer while black means faithful
    
#Finding the Frauds
mappings = som.win_map(X)
#we find all the mappings from the winning nodes to the customers
frauds = np.concatenate( (mappings[(8,1)],mappings[(8,2)]) , axis=0)

#here we put coordinates of the outliers in the frauds list to get the customers in that winning outlier
#axis=0->vertical axis

frauds = sc.inverse_transform(frauds)
























