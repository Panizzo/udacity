# Machine Learning Engineer Nanodegree
## Capstone Proposal
Daniel S. Panizzo 
February, 2020

## Proposal

### Domain Background

The usage of Neural Networks for image diagnosis is quickly growing in the field of medical research an its usage and efficiency are already being tested on clinics, laboratories and hospitals. With a variety of usage possibilities like cancer detection, classify lesion types or mental illness, the perspectives are enthusiastics. As I have an intense contact with doctors who work with diagnostic imaging and I frequently discuss the use of new technologies in the area, I intend to use this project as a starting point to delve deeper into the subject and perhaps participate in research projects in the area as Machine Learning engineer.

Reference: https://www.researchgate.net/publication/285912467_Artificial_Neural_Networks_in_Medical_Diagnosis

### Problem Statement

This project is based on the "Histopathologic Cancer Detection" Kaggle Competition, a project to identify metastatic tissue in histopathologic scans of lymph node sections. In others words, identify the presence of tumor cells in digital pathology scans. The expected results of this project is to train a Neural Network model to make a binary clssification of the presence of tumor (true or false) from a given image.

Reference: https://www.kaggle.com/c/histopathologic-cancer-detection/overview 

### Datasets and Inputs

In this dataset, we are provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. You are predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.

Reference: https://www.kaggle.com/c/histopathologic-cancer-detection/data 

### Solution Statement

The proposed solution is to use Convolutional Neural Networks to classify the pathology images as true or false for tumor tissue. We'll start exploring from the basic architecture of an CNN and see if we can improve the results applying pre-trained models (like Xception, VGG19 and NASNet) and fine tunning the parameters of the CNN. 

### Benchmark Model

As the main benchmark model, we'll use the notebook available from the user "CVxTz" in the Kaggle competition. His model got an score of 0.9709 using CNN with the NasNet pre-trained models. I should also use others competitors notebooks as benchmark for the exploratory data analysis and tests with others pre-trained models.

https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb

### Evaluation Metrics

The submissions for this competition are evaluated on area under the ROC curve between the predicted probability and the observed target. In the submission file, for each id in the test set, we must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format: 

id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.


### Project Design

EXPLORATORY DATA ANALYSIS (EDA)

We'll start exploring our data set to visualize the images we are using and how the features values are distributed.

BASE MODEL

After understanding our data, we'll build a basic CNN as a start point and evaluate it. This basic model should contain a couple of hidden layers with ReLu activation followed by a fully connected layer finished with Sigmoid activation. After the initial evaluation, we should iterate trough fine tuning the parameters and including some Drop Out layers to avoid overfitting in our model.

PRE-TRAINED MODELS 

Here we'll evaluate some of the pre-trained models available in Keras (https://keras.io/applications/). The main idea is to connect the output from the pre-trained model to our fully connected layer created in the previous stage and evaluate the results.

FINAL ANALYSIS

At the end, we'll compare the results and point the strengths and weaknesses observed in each model. At the end, we'll export our best model and submit it as kernel for the Kaggle competition.