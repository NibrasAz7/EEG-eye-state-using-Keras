import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from scipy import signal
from sklearn import preprocessing
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve,auc, precision_score,recall_score,f1_score
import pickle

#Dataset Reading
Data = pd.read_csv(r"D:\Master UNIVPM\Projects\01\eeg.csv") 

# Exploring Data
print(Data.dtypes)
print(Data.columns)
print("Data shape:",Data.shape)
print(Data.head())
print(Data.describe())
print(Data.info())
# Check for any nulls
print(Data.isnull().sum())
print(Data['eye'].describe())
print(Data['eye'].head())

Data['eye'].value_counts().plot.pie(labels = ["1-open","0-closed"],
                                              autopct = "%1.0f%%",
                                              shadow = True,explode=[0,.1])

#Data set Parameters
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'eye'] # channel names   
sfreq = 128  # sampling frequency, in hertz  
info = mne.create_info(ch_names, sfreq)

# Transfer Labels: From Open-Closse to 1-0 levels
Data['eye']=Data["eye"].astype('category')
Data["eye"] = Data["eye"].cat.codes

for i in ch_names[0:14]:
    plt.plot(Data[i])

# Despiking
for i in ch_names[0:14]:
    Data[i]=Data[i].where(Data[i] < Data[i].quantile(0.999), Data[i].mean())
    Data[i]=Data[i].where(Data[i] > Data[i].quantile(0.001), Data[i].mean())
    plt.plot(Data[i])
    
signal.detrend(Data[0:14], axis=- 1, type='linear', bp=0, overwrite_data=False)

for i in ch_names[0:14]:
    plt.plot(Data[i])
    
#Data Normalizing (min-max Scaling)
x = Data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Data = pd.DataFrame(x_scaled)
Data.columns=ch_names

for i in ch_names[0:14]:
    plt.plot(Data[i])

#Numpy Array to mne data
data = Data.to_numpy() # Pandas DataFrame to Numpy Array
data_mne = np.transpose(data)
raw = mne.io.RawArray(data_mne, info)   
raw.plot() #Plot all the signales

Correlation_df = Data.corr()

# Prepare Train and test Data
splitRatio = 0.3
train, test = train_test_split(Data ,test_size=splitRatio,
                               random_state = 123, shuffle = True)

train_X = train[[x for x in train.columns if x not in ["eye"]]]
train_Y = train["eye"]

feature_cols = train_X.columns

test_X = test[[x for x in train.columns if x not in ["eye"]]]
test_Y = test["eye"]

x_val = train_X[:1000]
partial_x_train = train_X[1000:]

y_val = train_Y[:1000]
partial_y_train = train_Y[1000:]

#Neural Network
## Building Model

model = models.Sequential()
model.add(layers.Dense(12, activation = 'relu', input_shape=(14,)))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1,  activation= 'sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#K.set_value(model.optimizer.learning_rate, 0.001)

## Training Model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=50,
                    batch_size=10,
                    validation_data=(x_val, y_val))

## Evaluating Model
### Network Architecture
print(model.summary())

### Training Process
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label="training loss")
plt.plot(epochs, val_loss_values, 'b', label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label="training acc")
plt.plot(epochs, val_acc_values, 'b', label="validation acc")
plt.title("Training and Validation acc")
plt.xlabel("Epoches")
plt.ylabel("acc")
plt.legend()
plt.show()

### Prediction
results = model.evaluate(test_X, test_Y)
print('Results: ', results)

predictions = model.predict(test_X)
for i in range(0,len(predictions)):
    predictions[i] = 1 if (predictions[i] > 0.5) else 0
Predictions = pd.DataFrame(predictions)
Predictions[0].value_counts().plot.pie(labels = ["1-open","0-closed"])
pd.value_counts(Predictions.values.flatten())

### Metrics
print("Accuracy:",accuracy_score(test_Y, predictions))
print("f1 score:", f1_score(test_Y, predictions))
print("confusion matrix:",confusion_matrix(test_Y, predictions))
print("precision score:", precision_score(test_Y, predictions))
print("recall score:", recall_score(test_Y, predictions))
print("classification report:", classification_report(test_Y, predictions))

### Plots
#### 01 plot Confusion Matrix as heat map
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y, predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#### 02 plot ROC curve
test_Y_01 =test_Y.astype('category')
test_Y_01 = test_Y_01.cat.codes

fpr,tpr,thresholds = roc_curve(test_Y, predictions)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=15)

# Saving Model
## Save the trained model as a pickle string. 
saved_model = pickle.dumps(model) 
  
## Load the pickled model 
NN_Model = pickle.loads(saved_model) 
  
## Use the loaded pickled model to make predictions 
NN_Model.predict(test_X) 