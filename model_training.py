
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib. image as mpimg
from tensorflow import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd. read_csv(os.path.join(datadir, 'driving_log.csv'),names = columns)
pd.set_option('display.max_colwidth' , -1)
data.head()

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data[ 'center'] = data['center'].apply(path_leaf)
data['left'] = data[ 'left'].apply(path_leaf)
data[ 'right'] = data['right'].apply(path_leaf)
data.head()

num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins [1:]) * 0.5
plt.bar(center, hist, width=0.05)

remove_list = []
for j in range(num_bins):
  list_=[]
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin: ]
  remove_list.extend(list_)
print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining.',len(data))

def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc [i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append (float(indexed_data[3]))
# left image append
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append (float(indexed_data[3])+0.15)
# right image append
    image_path.append (os.path.join(datadir,right.strip()))
    steering.append (float(indexed_data[3])-0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings= load_img_steering(datadir + '/IMG', data)

x_train, x_valid, y_train, y_valid= train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print("training samples: {}\nValidation Samples: {}".format(len(x_train),len(x_valid)))

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.5))
  image = zoom.augment_image(image)
  return image
def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.4))
    image = brightness.augment_image(image)
    return image
def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)

    return image, steering_angle

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
image = image_paths[550]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')

def batch_generator(image_paths, steering_ang, batch_size, istraining):

  while True:
    batch_img = []
    batch_steering = []

    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)

      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]

      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))

def nvidiaModel():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(64,(3,3),activation="elu"))
  #model.add(Convolution2D(64,(3,3),activation="elu"))
  #model.add(Dropout(0.5))

  model.add(Flatten())

  model.add(Dense(100,activation="elu"))
  #model.add(Dropout(0.5))

  model.add(Dense(50,activation="elu"))
  #model.add(Dropout(0.5))

  model.add(Dense(10,activation="elu"))
  #model.add(Dropout(0.5))

  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=1e-3),loss="mse", metrics=['accuracy'])

  return model

model = nvidiaModel()
print(model.summary())

history = model.fit_generator(batch_generator(x_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=50,
                                  validation_data=batch_generator(x_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

model.save('modelnew.h5')
from google.colab import files
files.download('modelnew.h5')