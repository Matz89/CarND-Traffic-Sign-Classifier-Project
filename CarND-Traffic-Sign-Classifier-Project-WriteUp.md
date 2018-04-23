
# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/Bar.png?raw=true
[image2]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/generated.png?raw=true
[image3]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/grayscale.png?raw=true
[image4]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/scaled_bar.png?raw=true
[image5]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/data/Extra_signs/class_11_rightofwayatnextintersection.jpg?raw=true
[image6]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/data/Extra_signs/class_17_noentry.jpg?raw=true
[image7]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/data/Extra_signs/class_28_childrencrossing.jpg?raw=true
[image8]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/data/Extra_signs/class_34_turnleftahead.jpg?raw=true
[image9]: https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/data/Extra_signs/class_9_nopassing.jpg?raw=true
[image10]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/28.png?raw=true
[image11]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/11.png?raw=true
[image12]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/9.png?raw=true
[image13]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/34.png?raw=true
[image14]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/17.png?raw=true
[image15]:https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Figures/randomSamples.png?raw=true

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Matz89/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

This is a bar chart showing the distribution among the class representations:


![alt text][image1]


And here is a random sample showing of an image for each class:

![alt text][image15]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces errors based on environment (ie. lighting, fading colour, etc).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because it reduces the amount of error that can occur, and helps with the gradient descent towards proper classification.

I decided to generate additional data because multiple classifications were under represented, and some were over represented.

To add more data to the the data set, I used the following techniques because increasing the dataset with slightly altered existing images is preferred to removed data from the over represented classes.

Here is an example of an original image and an augmented image:

![alt text][image2]

The difference between the original data set and the augmented data set is the following increase in the under represented classes (about 50% of the size of the most represented class).

![alt text][image4] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|         										|
| Max pooling			| 2x2 stride,  outputs 5x5x16         			|
| Flatten				|		Output 400								|
| Fully Connected		| Output 120									|
| RELU					|												|
| Dropout				| 82% for training set, 100% for everything else|
| Fully Connected		| Output 84										|
| RELU					|												|
| Dropout				| 82% for training set, 100% for everything else|
| Fully Connected		| Output 43										|
 	
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
| Hyper Parameter  		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate 		| 0.0011   										| 
| EPOCHS		     	| 27											|
| BATCH_SIZE			|	128											|
| dropout_keep_prob		|	0.82										|


To train the model, I initialized my biases with zeroes, and weights with a 0 mean and 1.0 standard deviation (defaults).  I followed up with cross entropy for calculating the accuracy.  Additionally I used the adam optimizer for improving the weight distribution when training the model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **.989**
* validation set accuracy of **.940**
* test set accuracy of **.917**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture as I felt it was applicable due to other examples handling similar information for training.  After first implementation, I felt it was necessary to continue as first results were positive, yet still able to be improved.

* What were some problems with the initial architecture?
*First validation results were fairly low, so attempts to identify a few areas was required.  Initial understanding for the large difference in class representation appears to not work well with this architecture.  Additionally, I find that a dropout mechanism would be required as it would help the model be developed when there might be some missing information, as to not overtune and also be able to handle a wider variety of possible images for classification.*

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

*I added dropout between the fully connected layers, as to allow the model to build general definitions for each class, and to not strongly tune into specific details that may not exist outside of the training set.  I did preprocess to generate additional images as to solve the under represented classes issue.*

* Which parameters were tuned? How were they adjusted and why?

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate 		| adjusted in small increments to reach higher accuracy  										| 
| EPOCHS		     	| adjusted in large increments to find plateaus, then fine tuned with smaller increments for increases in accuracy											|
| BATCH_SIZE			|	adjusted in powers of 2 to increase accuracy of model										|
| dropout_keep_prob		|	adjusted in small increments to increase accuracy of model									|
| Rotational Transform	|	adjusted in small increments to increase accuracy of model											|
| Class Rep Ratio		|	adjusted in large, then small, increments to find a comfortable ratio between training accuracy and validation accuracy											|



* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
*I found my choice for a dropout layer would be important, as signs can appear in many angles and lighting, so preventing a model from fitting in every detail, and to build general models, would be more beneficial for identifying the correct classification.*

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first image (Right-of-way at the next intersection) was a simply classification due to the basic background, lack of noise, and full shape.

The second image (No Entry) was a simple classification due to the clear shape, unimportant background, and decent lighting.

The third image (Children Crossing) would be a difficult classification due to the lighting, and the full shape is disturbed by some bushes/branches.

The fourth image (Turn left ahead) would be a difficult classification due to the angle of the sign in the image.

The fifth image (No Passing) would be a difficult classification due to the noisy background, and background being nearly the same colour as majority of the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing      		| Children Crossing   									| 
| Right-Of-Way    			| Right-Of-Way 										|
| No Passing| No Passing|
| Turn left ahead	      		| Turn left or straight ahead					 				|
| No Entry			| No Entry     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavourably to the accuracy on the test set of 91.7% as it is lower than expected.  The problem sign was the Turn Left Ahead, as it classified it as a Turn Left or Straight Ahead.  I suspect the main reason is due to the perspective of the image, as it is not dead-center as the other images have been.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Children Crossing sign (probability of 0.78), and the image does contain a Children Crossing sign. The top five soft max probabilities were:
![alt text][image10] 


For the second image, the model is quite sure that this is a Right-Of-Way at the next intersection sign (probability of 0.99), and the image does contain a Right-Of-Way at the next intersection sign. The top five soft max probabilities were:

![alt text][image11] 


For the third image, the model is quite sure that this is a No Passing sign (probability of 0.99), and the image does contain a No Passing sign. The top five soft max probabilities were:

![alt text][image12] 


For the fourth image, the model is relatively sure that this is a Turn Left or Straight Ahead sign (probability of 0.70), and the image does contain a Turn Left Ahead sign.  This was incorrectly classified by the model. The top five soft max probabilities were:

![alt text][image13] 


For the fifth image, the model is quite sure that this is a No Entry sign (probability of 0.99), and the image does contain a No Entry sign. The top five soft max probabilities were:

![alt text][image14]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

