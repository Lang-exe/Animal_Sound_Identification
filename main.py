# Import Library
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf

# audio plot
from scipy.io import wavfile as wav
import IPython.display as ipd

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense,GlobalAveragePooling2D, Dropout
from sklearn.metrics import confusion_matrix, classification_report

# Load Data
!unzip '/content/Dataset.zip' -d './dataset'

# Create Dataset Path
dataset_path = '/content/dataset/train'

# Gathering Data
dataset = list(glob.glob(dataset_path + '/**/*.wav'))

labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], dataset))
file_path = pd.Series(dataset, name = 'File_Path').astype(str)
labels = pd.Series(labels, name = 'Labels')
data = pd.concat([file_path, labels], axis = 1)
data = data.sample(frac = 1).reset_index(drop = True)
data.head()

# Explanatory Data Analysis (EDA) and Visualization
# Show Data
data.head()

# Show Total Sample of Each Class
print('TOTAL SAMPLE OF EACH CLASS')
print(data['Labels'].value_counts())
sns.countplot(x = 'Labels', data = data)
plt.title('Total Sample of Dog and Cat Class')
plt.show()

# Show Waveform Sample from Data
for i in range(4):
    x, sr = librosa.load(data.File_Path[i])
    plt.figure(figsize = (10, 5))
    plt.title('Label : ' + str(data['Labels'][i]))
    librosa.display.waveshow(x, sr = sr)

# Display Audio
ipd.Audio(data.File_Path[1])

# Display Audio
ipd.Audio(data.File_Path[4])

# Data Preprocessing
# Feature Extraction - MFCC
def feature_extract(file_name):
    audio, sr = librosa.load(file_name, res_type='soxr_vhq')
    mfccs = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = 40)
    mfccs_processed = np.mean(mfccs.T, axis = 0)
    return mfccs_processed

features = []

for index, row in data.iterrows():
    file_name = str(row['File_Path'])
    class_label = row["Labels"]
    data_ = feature_extract(file_name)
    features.append([data_, class_label])

features_df = pd.DataFrame(features, columns = ['Feature', 'Class_Label'])
features_df.head()

#Separate Variable
x = np.array(features_df.Feature.tolist())
y = np.array(features_df.Class_Label.tolist())

#Label Encoder
label = LabelEncoder()
y = to_categorical(label.fit_transform(y))

#Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)

# Modelling Using CNN
model = Sequential()
model.add(Dense(128, input_dim = 40,  activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

#Training Model
history = model.fit(
    x_train,
    y_train,
    epochs = 100,
    batch_size = 64
)

# Evaluation Model
score = model.evaluate(x_train, y_train, verbose = 0)
print("Training Model Accuracy : \033[01m {0:.2%}\033[0m".format(score[1]))

# Prediction
# Create Dataset Path
testset_path = '/content/dataset/train'
testset = list(glob.glob(testset_path + '/**/*.wav'))

labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], testset))
file_path = pd.Series(dataset, name = 'File_Path').astype(str)
labels = pd.Series(labels, name = 'Labels')
data = pd.concat([file_path, labels], axis = 1)
data = data.sample(frac = 1).reset_index(drop = True)
data.head()

def prediction(path_sound):
    data_sound = feature_extract(path_sound)
    X = np.array(data_sound)
    X = X.reshape(1, 40)
    pred_ = model.predict(X)
    pred_ = np.argmax(pred_, axis = 1)
    pred_class = label.inverse_transform(pred_)
    print("The predicted class : \033[01m ", pred_class[0],'\033[0m \n')

path_sound = '/content/dataset/train/cat/cat_1.wav'
prediction(path_sound)
ipd.Audio(path_sound)

path_sound = '/content/dataset/train/dog/dog_barking_1.wav'
prediction(path_sound)
ipd.Audio(path_sound)

y_true = []
y_pred = []

def conf_matrix(path_sound):
    data_sound = feature_extract(path_sound)
    X = np.array(data_sound)
    X = X.reshape(1, 40)
    pred_ = model.predict(X)
    pred_ = np.argmax(pred_, axis = 1)
    pred_class = label.inverse_transform(pred_)
    return pred_class[0]

for index, row in data.iterrows():
    file_name = str(row['File_Path'])
    class_label = row["Labels"]
    data_ = conf_matrix(file_name)
    y_true.append(data_)
    y_pred.append(class_label)

print(confusion_matrix(y_true, y_pred, labels = ['cat', 'dog']))
print(classification_report(y_true, y_pred, labels = ['cat', 'dog']))
