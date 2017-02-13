import pickle
import math
import random
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import initializations
from keras.utils.visualize_util import plot
from pathlib import Path

import json
import tensorflow as tf
tf.python.control_flow_ops = tf  

#Read lines from csv lof file
with open('driving_log.csv', mode='r') as f:
    lines = f.readlines()

images = []
angles = []

#parameters
img_row = 160
img_col = 320
channels = 3
b_size = 32 
n_epochs = 3
n_train_samples = b_size*750 
n_val_samples = int(n_train_samples*0.2/0.8)

#Data augmentation functions
#ImageFlip
def vertical_flip(image,angle):
    image = cv2.flip(image,1)
    angle = -1.0*angle
    return image, angle

#Horizontal random traslation
def traslation(image,angle):
    xtraslation = 100*np.random.uniform()-100/2
    angle_traslation = angle + xtraslation*.0016
    ytraslation = 40*np.random.uniform()-40/2
    Traslation_Matrix = np.float32([[1,0,xtraslation],[0,1,ytraslation]])
    image_traslation = cv2.warpAffine(image,Traslation_Matrix,(image.shape[1],image.shape[0]))
    return image_traslation,angle_traslation

#Random brightness generator
def brightness(image,angle):
    image = cv2.cvtColor(image,cv2.COLOR_YUV2BGR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    angle = angle*1.0
    return image,angle

#Image Random Generator (image.shape => (40,80,3)) 
def generator(image,angle):
    #Randomly choose the type of preprocessing for each image
    step = np.random.randint(6)
    if(step == 0):
        image, angle = vertical_flip(image,angle)
    elif(step == 1):
        image, angle = traslation(image,angle)
    elif(step == 2):
        image, angle = vertical_flip(image,angle)
        image, angle = traslation(image,angle)
    elif(step == 3):
        image, angle = brightness(image,angle)
        image, angle = vertical_flip(image,angle)
        image, angle = traslation(image,angle)
    elif(step == 4):
        image, angle = brightness(image,angle)
        image, angle = traslation(image,angle)
    else:
        image, angle = brightness(image,angle)
    resize = cv2.resize(image,(int(img_col/4.0),int(img_row/4.0)),interpolation=cv2.INTER_AREA) 
    return resize, angle

#Batch generator from CSV File - n batches of 32 images with shape = (40,80,3)
def batch_generator(data,angles,mode,batch_size = 32):
    batch_images = np.ndarray(shape=(batch_size,int(img_row/4.0),int(img_col/4.0),channels), dtype=np.uint8)
    batch_angles = np.zeros(batch_size,)
    while 1:
        for i in range(batch_size):
            index = random.randint(0, len(data)-1)
            #load random image from CSV file
            image = cv2.imread(data[index],cv2.IMREAD_COLOR)
            angle = angles[index]
            #change space color to YUV
            image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
            #image cropping
            image = image[math.floor(image.shape[0]/4):image.shape[0]-25, 0:image.shape[1]]
            #mode 0 is for training images generator - mode 1 is for validation images generator
            if(mode==0):
                image, angle = generator(image,angle)
                batch_images[i] = image
                batch_angles[i] = angle
            if(mode==1):
                batch_images[i] = cv2.resize(image,(int(img_col/4.0),int(img_row/4.0)),interpolation=cv2.INTER_AREA)
                batch_angles[i] = angle
            image, angle = generator(image,angle)
            batch_images[i] = image
            batch_angles[i] = angle
        batch_images, batch_angles = shuffle(batch_images, batch_angles)
        yield batch_images, batch_angles

#Model definition    
def regression_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(int(img_row/4.0),int(img_col/4.0), channels), output_shape=(int(img_row/4.0),int(img_col/4.0), channels)))
    model.add(Convolution2D(1,1,1, border_mode = 'same',init ='glorot_uniform',name='conv1'))
    model.add(Convolution2D(32, 5, 5, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv2'))
    model.add(Convolution2D(32, 5, 5, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv3'))
    model.add(MaxPooling2D((2, 2),strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv4'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv5'))
    model.add(MaxPooling2D((2, 2),strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv6'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid',activation='elu',init='glorot_uniform',W_regularizer=l2(0.),name='conv7'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, init='glorot_uniform', W_regularizer=l2(0.)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='glorot_uniform', W_regularizer=l2(0.)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, init='glorot_uniform', W_regularizer=l2(0.)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    #adaptive optimizer -> ADAM with mean squared error
    model.compile(optimizer='adam', loss='mse')
    return model

#remove header 
lines = lines[1:len(lines)]

#Parser - Add (0.1 |-0.1) to (left | right) steering angles  
for line in lines:
    blocks = line.split(',')
    # Center image
    images.append(blocks[0])
    angles.append(float(blocks[3]))
    # Left image
    images.append(blocks[1][1:])  
    angles.append(float(blocks[3])+0.1)
    # Right image
    images.append(blocks[2][1:])  
    angles.append(float(blocks[3])-0.1)
    
X_data = np.array(images)
y_data = np.array(angles)

#80% training set - 20% validation set
X_data, y_data = shuffle(X_data, y_data)
X_data, X_validation, y_data, y_validation = train_test_split(X_data, y_data, test_size = 0.20)

#Model instantiation
model = regression_model()
model.summary()
plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)

#Training and validation
X_data, y_data = shuffle(X_data, y_data)
X_validation, y_validation = shuffle(X_validation, y_validation)
history = model.fit_generator(batch_generator(X_data, y_data, 0, b_size),
                              n_train_samples,
                              n_epochs,
                              validation_data=batch_generator(X_validation, y_validation, 1, b_size),
                              nb_val_samples=n_val_samples)

#convert model to json
json_string = model.to_json()

#remove old versions
try:
    os.remove('model.json')
    os.remove('model.h5')
except OSError:
    pass

#save the model
with open('model.json', 'w') as f:
    f.write(json_string)
model.save_weights('model.h5')

#save history (training and validation loss)
with open('history.p', mode='wb') as f:
    data = pickle.dump(history.history,f)

gc.collect()
K.clear_session()

print('Done.')
