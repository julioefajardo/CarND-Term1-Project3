#**Behavioral Cloning with Keras** 
---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/Dataset.png "Udacity Dataset"
[image3]: ./img/Augmented.png "Augmented Images (Flipped and Traslated Images)"
[image4]: ./img/Distribution.png "Steering Angles Distribution Before Data Generation"
[image5]: ./img/History.png "Train and Validation Loss"
[image6]: ./img/Simulation_2.png "Autonomous Driving Test"
[image7]: ./img/Simulation_3.png "Autonomous Driving Test"

## Rubric Points
###Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model.json needed for driving the car in autonomous mode 
* Report.md summarizing the results

Since, the new beta simulator does not run in my virtual machine, i used the default simulator to test my model. Using the Udacity provided simulator (default) and my drive.py file, the car can be driven autonomously around the track by executing. 
```sh
python drive.py model.json
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Since, you have been noted the simulator might perform differently based on the hardware, i provided a video below.

###Model Architecture

My model consists of a convolution neural network, with 7 convolutional layers, max-pooling and 3 fully conected layers. The size of the kernels varies according to the convolution layer, with 5x5, 3x3 and 1x1 filter sizes and depths between 32 and 128 (model.py lines 118-146). The first layer (Keras Lambda) was used to normalize the data between -0.5 to 0.5, the second layer is composed of 3 filters of size 1X1, in order to let the system choose the best color space of the images.  This is followed by 2 convolutional blocks each composed of 32 filters of size 5X5, followed by a Max-Pooling layer with a (2,2) stride. Then, is followed by another 2 convolutional blocks each composed of 64 filters of size 3X3, followed by a Max-Pooling layer with a (2,2) stride. Then, is followed by another 2 convolutional blocks each composed of 128 filters of size 3X3. These convolution layers are followed by 3 fully connected layers with 128, 64 and 16 sizes respectively. I choose 'elu' activation for each layer, in order to get smoother steering angles in the output. Also, each layer was initialiazed with method proposed by X. Glorot in http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf. This, with the aim of get faster convergence. Moreover, the model contains Dropout layers and L2 Weight Regularizaton in order to reduce overfitting. The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 145).


```sh
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
```
Here is a visualization of the architecture:

![alt text][image1]

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 168-169). Also, for training dataset the images were processed on different manners in order to get fake data to avoid overfitting (model.py lines 67-87). For validation dataset the images only were resized, in order to validate the model with real situations taken from the simulator (model.py 104-110). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Training data was chosen from the dataset provided by Udacity. Since, the simulator was a poor performance in my virtual machine, drive around the track was a nightmare using the keyboard (i am not a gamer, so i do not have joystikc), so not useful data was collected from that attempts.

####Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image3]


Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...

![alt text][image4]

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

![alt text][image5]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image6]
![alt text][image7]
