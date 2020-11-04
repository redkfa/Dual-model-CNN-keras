# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:30:43 2018

@author: s207
"""

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
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# dimensions of our images.
img_width, img_height = 50,50

train_data_dir = 'train/pc1'
validation_data_dir = 'test/pc1'
nb_train_samples = 622
nb_validation_samples = 162
epochs = 10000
batch_size = 32

steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size
num_classes = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


Color_model = Sequential()
Color_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
Color_model.add(Conv2D(64, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(64, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(32, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(32, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Flatten())
Color_model.add(Dense(128, activation='relu'))
Color_model.add(Dense(num_classes, activation='softmax'))



#compile the model
Color_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=False)
tensorboard.set_model(Color_model)
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

Color_model.fit_generator( 
                                train_generator,
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              use_multiprocessing=True,
                              shuffle=False,verbose=2
                              ,  callbacks=[tensorboard] )

Color_model.summary() 
#Confution Matrix and Classification Report
Y_pred = Color_model.predict_generator(validation_generator, nb_validation_samples/batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['block', 'cup', 'glasses', 'key', 'pill', 'smartphone', 'usb', 'wallet']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#head_model.save_weights('resnet_classification.h5')
Color_model.save('self_solo50_classification_model.h5')
