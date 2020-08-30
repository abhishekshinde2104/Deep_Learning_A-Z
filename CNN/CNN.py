#Part 1:

#we wont use data preprocessing 
#we just use feature scaling and image augmentation

#Part 2: Building the CNN
from keras.models import Sequential#initialise NN
from keras.layers import Convolution2D#1st step of CNN 2D for images CNN layer
from keras.layers import MaxPooling2D#2nd step pooling layers
from keras.layers import Flatten#3rd step Flattened layer
from keras.layers import Dense#used yo connect CNN to ANN

#Initialising CNN
classifier=Sequential()

#Step 1 - Convolution
"""here we apply different feature layers on the image and get different feature maps
"""
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#32->no of feature detectors = feature maps
#3->no of rows in feature detector
#3->no of columns in feature detector
#input_shape -> shape of your input image on which u are going to apply convolution operation
#as all images dont have same format 
#order of input_shape = 64 x 64 ->size of 2D array and 3->RGB channels
#we use activation function relu just to make sure we dont have any negative pixel values in our feature map
#we need to remove negative pixels to remove the linearity 

#Step 2 - Pooling 
#we reduce the size of feature maps to reduce the number of fully connected layers
classifier.add(MaxPooling2D(pool_size=[2,2]))
#pool_size is the stride from where we take the maximum number among the pixels

"""ADDING CONVOLUTIONAL LAYER AFTER 1ST CONVOLUTINAL LAYER AND AFTER MAXPOOLING"""
#2nd Convolutional layer
#here input isnt the images but will be the pooled feature maps from previous layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=[2,2]))

#Step 3 - Flattening
#this pulls out all the pooled feature maps and converts them into 1 single vector 
#and this single layer is going to be input layer of ANN
"""Q1)WHY DONT WE LOSE THE SPATIAL STRUCTURE BY FLATTENING ALL THESE FEATURE MAPS INTO ONE SAME SINGLE VECTOR
Ans 1) Thats bcoz by creating our feature maps we extracted the sptial struture informations
by getting those high numbers in the feature maps by the feature detectoe we applied on the 
input image through convolutional step
So basically these high numbers represent the spatial structure of our images bcoz the 
high numbers in the feature maps are associated to a specific feature in the input image 
and by applying max_pooling step we keep these high numbers and flattening steps just makes it into
1 single vector we still have the high numbers 
and since these high numbers represent the spatial structure of the input image and are 
associated to 1 specific feature of the spatial structure we keep this information

Q2)WHY DIDNT WE DIRECTLY TAKE ALL THE PIXELS OF THE INPUT IMAGE AND FLATTEN THEM INTO THIS ONE SAME SINGLE VECTOR
Ans 2 )The 1D vector would be very huge and there wont be any relation between the pixels 
so we dont get any information"""

classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(output_dim=128,activation='relu')) #hidden layer

classifier.add(Dense(output_dim=1,activation='sigmoid')) #binary outcome so sigmoide

#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Image preproceesing Step
#Part 3 -Fitting the CNN to the images
#we will use image augmentatioin to avoid overfitting
#if we dont do image augmentation then we will get a higher accuracy result on train set 
#but a lower accuracy result on test set
"""Before fitting cnn to train set we do image augmentatio to avoid overfitting"""
from keras.preprocessing.image import ImageDataGenerator
"""Q)what is image augmentation and how will it prevent overfitting
==>One of the situations that lead to overfitting is when we have few data to train our model
and in that situation our model finds some correlations in the few observations of the traininng set
but fails to generalize these correlations on some new observations
And when it comes to images we actually need a lot of images to find and generalize some correlations
bcoz in computer vision our ML model doesnt simply need to find some correlations between some 
independent variable and some dependent variables it also needs to find some patterns in pixels
and to do it requires a lot of images.
And hence we use images augmentation as it creates many batches of our images and then for each
batch it will apply some random transformation on a random selection of our image just like
rotation,flipping,shifting,shearing etc. and eventually we get many more diverse images during 
the training and therefore a lot more material to train.
Hence image augmentation avoids overfitting
SUMMARY : IMAGE AUGMENTATION IS A TECHNIQUE THAT ALLOWS US TO ENRICH OUR DATASETS OUR TRAINSET
WITHOUT ADDING MORE IMAGES AND THEREFORE THAT ALLOWS US TO GET GOOD PERFORMANCE RESULTS WITH 
LITTLE OR NO OVERFITTING.
"""

# Generating images for the Training set
#rescale is compulsory
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""this object we will use to augment the images of the train set"""


# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
#another object used to preprocess the test set

training_set=train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')
"""here we apply image augmentation itself on the images of our train set at the same time
resizing all our images to 64,64 and create batch of 32 and then our cnn will be trained on this set"""

#target_size ->size of images expected in the cnn model 
#but as above chose input_shape=64,64 so here also we need to keep it 64,64
#batch_size->size of the batches in which some random samples of our images will be included
#and that contains the no of images that will go through CNN after which weights will be updated
#class_mode->this parameter indicates if your dependent variable is binary or has more than 2 categories
 

test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')


classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)
#fit_generator will not only fit cnn to the training set but it will also test at the same time its
#performance on some new observations which are gonna be obs of test set
#sample_per_epoch-->no of images on trainig set 
#nb_epoch-->number of rounds to train our cnn
#nb_val_samples-->number of images in test set

"""Acuracy of train set == 85%
Acuracy of test set = 75%
but as this difference is much large we need to decrease it 
1)by either adding another convolutional layer
2)add another fully connected layer

#here we adding another convolutional layer"""
"""Higher accuracy can also be achieved by increasing the target_size that will increase the pixel pattern
and hence more features and hence more information."""


















