from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization

# dimensions of our images.
img_width, img_height = 256,256

train_data_dir = 'train/pc1'
validation_data_dir = 'test/pc1'
nb_train_samples = 622  #7331
nb_validation_samples = 162
epochs = 100
batch_size =4
steps_epoch = nb_train_samples/batch_size
validation_steps= nb_validation_samples/batch_size

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (256,256,3))

#number of classes in your dataset e.g. 20
num_classes = 8

x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

#create graph of your new model
head_model = Model(input = base_model.input, output = predictions)

#compile the model
head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


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

head_model.fit_generator(
    train_generator,
    steps_per_epoch=steps_epoch ,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps )

head_model.save_weights('pc1_classification.h5')
head_model.save('pc1_classification_model.h5')