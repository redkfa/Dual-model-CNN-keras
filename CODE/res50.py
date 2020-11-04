# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:16:23 2018

@author: s207
"""
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# dimensions of our images.
img_width, img_height = 224,224

train_data_dir = 'train/pc1'
validation_data_dir = 'test/pc1'
nb_train_samples = 622
nb_validation_samples = 162
epochs = 10000
batch_size = 16

steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size
num_classes = 8
'''
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
'''

base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))

#number of classes in your dataset e.g. 20
num_classes = 8

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

#create graph of your new model
head_model = Model(input = base_model.input, output = predictions)


#compile the model
head_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=False)
tensorboard.set_model(head_model)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
#checkpoint = ModelCheckpoint('model_pc0_bestResnet_224.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

head_model.fit_generator( 
                                train_generator,
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              use_multiprocessing=True,
                              shuffle=False,verbose=2
                              ,  callbacks=[tensorboard] )
head_model.summary() 
#Confution Matrix and Classification Report
Y_pred = head_model.predict_generator(validation_generator, nb_validation_samples/batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['block', 'cup', 'glasses', 'key', 'pill', 'smartphone', 'usb', 'wallet']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#head_model.save_weights('resnet_classification.h5')
head_model.save('resnet_classification_model.h5')
