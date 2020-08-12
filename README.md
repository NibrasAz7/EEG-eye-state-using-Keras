# EEG-eye-state-using-Keras

# Introduction
## Goal: 
Determain the eye state (Open/Close) based on EEG signal Using Deep Neural Network (Keras)

## Dataset
### Dataset Source:
Oliver Roesler, it12148 '@' lehre.dhbw-stuttgart.de , Baden-Wuerttemberg Cooperative State University (DHBW), Stuttgart, Germany
https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State

### Dataset Details:
All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. The duration of the measurement was 117 seconds. The eye state was detected via a camera during the EEG measurement and added later manually to the file after analysing the video frames. '1' indicates the eye-closed and '0' the eye-open state. All values are in chronological order with the first measured value at the top of the data.

### Features of Emotiv EEG Neuroheadset:
14 channels
Rigid Electrode Placement (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
Wet Electrodes (require saline water)
2048 Hz Sampling Rate
14 bits

Note: Measurment duration is 117 seconds with 14980 sample point. Therfore, Sampling Frequency equels to: 14980/117 = 128 Hz.
The data was downsampled from 2048 Hz to 128

# Methodology

## Exploring Data
This step aims to understand the data in hand.
We can notice that the data contains 45% of samples while eyes are closed while the other 55% of samples when the eyes were close.
By plotting the EEG signales, we can notice number of spikes (outliers) in the data.

## PreProcessing
1- Remove Spikes: Remove all the random numbers that lie in the lowest quantile (0.1%) and the highest quantile (99.9%).
2- Remove Trend (DC current shifting).
3- Normalization using Min-Max Scaling.

## Prepearing data
Spliting the data into training data and testing data, each data contains Featuers and Labels
The featers where considered as the amplitude of the signales at each sample.
Spit ratio were choosen to be 30%.

## Traiing Models
Model: "sequential_1"

optimizer = 'rmsprop'

loss = 'binary_crossentropy'

=================================================

Layer (type) - Output Shape - Param - Activation

========================================

dense_1 (Dense) - None, 12) - 180 - relu

========================================

dense_2 (Dense) - (None, 8) - 104 - relu

========================================

dense_3 (Dense) - (None, 4) - 36 - relu

========================================

dense_4 (Dense) - None, 1) - 5 - softmax

========================================


Total params: 325

Trainable params: 325

Non-trainable params: 0

# Results
## 1- Decision Tree (DT)
Accuracy:         0.8297730307076101

f1        score:  0.852116760100522
            
precision score:  0.8165987402741757

recall    score:  0.890864995957962

confusion matrix: [[1525  495]
                   [ 270 2204]]

# Descussion
We can notice that KNN performe better than DT.
Classification Performance is good in general.

# Futuer Work Suggestion
The daata could be segmented into opening sessions and clossing session.
It is suggested that more profound featuers could be calculated from those sessions.

## Resources:
Roesler, O. (2013). EEG Eye State Data Set. UCI Machine Learning Repositorty. Avilable on: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State. Last Accesss [06 Aug 2020].
Consumer EEG Headsets. (N.A.). Avilable on: http://learn.neurotechedu.com/headsets/. Last Accesss [06 Aug 2020].
