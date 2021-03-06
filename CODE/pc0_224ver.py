# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 05:13:27 2018

@author: s207
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 02:43:10 2018

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
img_width, img_height = 224,224

train_data_dir = 'train/pc1'
validation_data_dir = 'test/pc1'
nb_train_samples = 622  #7331
nb_validation_samples = 162
epochs = 10000
batch_size =32
steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size
num_classes = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    


Color_model =  Sequential()
Color_model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
Color_model.add(Conv2D(64, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(64, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(128, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))


Color_model.add(Conv2D(128, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(64, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))
Color_model.add(Conv2D(32, (3, 3), activation='relu'))
Color_model.add(MaxPooling2D(pool_size=(2, 2)))
Color_model.add(Dropout(0.25))

Color_model.add(Flatten())
Color_model.add(Dense(128, activation='relu'))
Color_model.add(Dropout(0.5))
Color_model.add(Dense(num_classes, activation='softmax'))


inp = Color_model.input
out = Color_model.output


Depth_model =  Sequential()
Depth_model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
Depth_model.add(Conv2D(64, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))
Depth_model.add(Conv2D(64, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))
Depth_model.add(Conv2D(128, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))


Depth_model.add(Conv2D(128, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))
Depth_model.add(Conv2D(64, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))
Depth_model.add(Conv2D(32, (3, 3), activation='relu'))
Depth_model.add(MaxPooling2D(pool_size=(2, 2)))
Depth_model.add(Dropout(0.25))


Depth_model.add(Flatten())
Depth_model.add(Dense(1024, activation='relu'))
Depth_model.add(Dropout(0.5))

Depth_model.add(Dense(128, activation='relu'))
Depth_model.add(Dropout(0.5))
Depth_model.add(Dense(num_classes, activation='softmax'))



for layer in Depth_model.layers:
    layer.name = layer.name + str("two")
inp2 = Depth_model.input
out2 = Depth_model.output

concatenated = concatenate([out,out2], axis=-1)
allout = Dense(num_classes, activation = 'softmax',name = 'theout')(concatenated)


self_model = Model([inp, inp2], allout)

'''
self_model=load_model('pc0_classification_model_224.h5')
'''
color_train_datagen = ImageDataGenerator( rescale=1. / 255 ,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
color_test_datagen = ImageDataGenerator(rescale=1. / 255)

#compile the model
self_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=False)
tensorboard.set_model(self_model)

def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                         
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=1)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label



inputgenerator = generate_generator_multiple(generator=color_train_datagen,
                                             dir1='train/pc1',
                                             dir2='train/pc2',
                                             batch_size=batch_size,
                                            # classes=['block', 'cup', 'glasses', 'key', 'pill', 'smartphone', 'usb', 'wallet'],
                                             img_height=img_height,
                                             img_width=img_height)

val_generator = generate_generator_multiple(color_test_datagen,
                                            dir1='test/pc1',
                                            dir2='test/pc2',
                                            batch_size=batch_size,
                                           # classes=['block', 'cup', 'glasses', 'key', 'pill', 'smartphone', 'usb', 'wallet'],
                                            img_height=img_height,
                                            img_width=img_height)

testgenerator = color_test_datagen.flow_from_directory('test/pc1',
                                          target_size=(img_height, img_width),                                      
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=1)
                        
testgenerator2 = color_test_datagen.flow_from_directory('test/pc2',
                                          target_size=(img_height, img_width),                                      
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=1)                        
checkpoint = ModelCheckpoint('model_pc0_best_224.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

history = self_model.fit_generator(inputgenerator,
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=val_generator,
                              validation_steps=validation_steps,
                              use_multiprocessing=True,
                              shuffle=False,verbose=2,
                              callbacks=[checkpoint,tensorboard])
self_model.summary() 
#Confution Matrix and Classification Report
Y_pred = self_model.predict_generator(val_generator, nb_validation_samples/batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(testgenerator.classes, y_pred))
print('Classification Report')
target_names = ['block', 'cup', 'glasses', 'key', 'pill', 'smartphone', 'usb', 'wallet']
print(classification_report(testgenerator.classes, y_pred, target_names=target_names))


                                    
#self_model.save_weights('pc0_classification.h5')
self_model.save('pc0_classification_model_224.h5')
