from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
# dimensions of our images.
img_width, img_height = 224,224

train_data_dir = 'C'
validation_data_dir = 'C'
Dtrain_data_dir = 'D'
Dvalidation_data_dir = 'D'
nb_train_samples = 134  #7331
nb_validation_samples = 12
epochs =1000
batch_size =1
steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size
num_classes = 8


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
    
    

Color_model = load_model('pc1_classification_model_224.h5')
Color_model.load_weights('pc1_classification_224.h5')
inp = Color_model.input
out = Color_model.output


Depth_model = load_model('pc1_classification_model_224.h5')
Depth_model.load_weights('pc2_classification_224.h5')
for layer in Depth_model.layers:
    layer.name = layer.name + str("two")
inp2 = Depth_model.input
out2 = Depth_model.output

concatenated = concatenate([out,out2], axis=-1)
allout = Dense(num_classes, activation = 'softmax',name = 'theout')(concatenated)


self_model = Model([inp, inp2], allout)
Color_model.trainable = False
Depth_model.trainable = False
'''
self_model = load_model('7_classification_model.h5')
'''
self_model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True, write_images=False)
tensorboard.set_model(self_model)

color_train_datagen = ImageDataGenerator( rescale=1. / 255 ,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
color_test_datagen = ImageDataGenerator(rescale=1. / 255)



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
                                             img_height=img_height,
                                             img_width=img_height)

testgenerator = generate_generator_multiple(color_test_datagen,
                                            dir1='test/pc1',
                                            dir2='test/pc2',
                                            batch_size=batch_size,
                                            img_height=img_height,
                                            img_width=img_height)

history = self_model.fit_generator(inputgenerator,
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=testgenerator,
                              validation_steps=validation_steps,
                              use_multiprocessing=True,
                              shuffle=False,verbose=2,
                              callbacks=[tensorboard])






'''

# this is the augmentation configuration we will use for training
depth_train_datagen = ImageDataGenerator( rescale=1. / 255,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

depth_test_datagen = ImageDataGenerator(rescale=1. / 255)



color_train_generator = color_test_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

color_validation_generator = color_test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)


depth_train_generator = depth_train_datagen.flow_from_directory(
    Dtrain_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size

)

depth_validation_generator = depth_test_datagen.flow_from_directory(
    Dvalidation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size

)


history = self_model.fit_generator(generator =[color_train_generator,depth_train_generator],
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=[color_validation_generator,depth_validation_generator],
                              validation_steps=validation_steps
                              )
'''


self_model.save_weights('7_classification.h5')
self_model.save('7_classification_model.h5')