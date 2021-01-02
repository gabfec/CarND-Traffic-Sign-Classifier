# **Traffic Sign Recognition** 


### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/samples_classes.png "Samples"
[image2]: ./writeup/samples_distribution.png "Distribution"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./signs-pics/1.jpg "Traffic Sign 1"
[image5]:  ./signs-pics/2.jpg "Traffic Sign 2"
[image6]:  ./signs-pics/3.jpg "Traffic Sign 3"
[image7]:  ./signs-pics/4.jpg "Traffic Sign 4"
[image8]: ./signs-pics/5.jpg "Traffic Sign 5"


### Data Set Summary & Exploration

Summary statistics of the traffic signs data set are computed after loading the pickle files in python:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is a view with one example of each of the traffic sign classes:

![alt text][image1]

Here is a bar chart showing samples distribution per class:

![alt text][image2]

This shows that the data set is not uniformly distributed, some traffic signs having more samples than the others. 

### Design and Test a Model Architecture

The images in the data set are color, 32x32 pixels. Before feeding the training model, they are preprocessed with 2 steps:
 - convert to greyscale
 - normalize in order to get a mean value of 0 and to reduce variance 


My final model is based on LeNet and  consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   						| 
| #1 Convolution 5x5   |  1x1 stride, valid padding,, outputs 28x28x6 	|
| Activation		| RELU										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| #2 Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x16 	|
| Activation		| RELU									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		      	| Outputs 400 				|
| Dropout		      	| 50% keep 				|
| #3 Fully connected	| output 120								|
| Activation		| RELU									|
| #4 Fully connected	| output 84							|
| Activation		| RELU									|
| #5 Softmax		| output 43								|

 
To train the model, I defined the following hyperparameters:
 - learning step for the Adam optimizer of 0.001
 - number of EPOCHS = 35
 - BATCH_SIZE = 128
 - dropout keep = 50%

I stated with the original LeNet architecture, but since I couldn't achive more than 0.91 accuracy I started to experiment with learning rate and the number of epochs. Finally what worked out was adding a dropout step.

My final model results were:
* validation set accuracy of 0.954
* test set accuracy of 0.938
 

### Test the Model on New Images

Here are five German traffic signs taken from the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

These were high resolution and bigger pictures and have been cropped to reduce the resolution and to make them look like to the ones used for learning.

The first two images might be more challenging to classify since, after lowering the resolution to 32x32, the details from the center of the image will be faded out.


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice/snow      		| Beware of ice/snow  									| 
| Pedestrians     			| Pedestrians										|
| Stop					| Stop											|
| Speed limit (60km/h)	      		| Speed limit (60km/h)				 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess all of the traffic signs, with an accuracty of 100%. 

For printing the first 5 softmax probabilities I truncated to 3 decimals.

|      | #1  | #2 | #3 | #4 | #5 | 
|:----|:----:|:--:|:----:|:-----:|:----| 
|Beware of ice/snow|0.99 | 0.009 | 0.001 | 0 | 0 |
|Pedestrians|0.861 | 0.139 | 0.001| 0 | 0 |
|Stop|1 | 0 | 0 |0 |  0|
|Speed limit (60km/h)|1 | 0 | 0 | 0 |  0|
|Speed limit (30km/h)|1 | 0 | 0 | 0 |  0|

These probabilities are close to the validation accuracty.
For the second image, the probability is lower and this was expected. The rest of the images are surprisingly classified with 100% confidence.
The original good quality of the image was, however, a factor for these good results. These prbabilities vary from execution to execution, but most of the times are close to 1.
