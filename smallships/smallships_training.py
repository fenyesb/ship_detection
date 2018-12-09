from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

img_width, img_height = 224, 224 # dimensions of the images.

#image folder
train_data_dir = 'smallships/train'
validation_data_dir = 'smallships/val'
#number of samples
nb_train_samples = 26782
nb_validation_samples = 5778
epochs = 20 #for first phase
batch_size = 16

def preprocess_input(img): #using the same preprocessor as VGG16
    # 'RGB'->'BGR'
    img = img[:, :, ::-1]
    # Zero-center by mean pixel
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img
    

# prepare data augmentation configuration:
# Data augmentation is used for creating more training samples
# (since it's low count) by adding shear, flip and zoom
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1. / 255, #scale the input (scale from 0-255 => 0-1)
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# on the validation and test files only perform scaling
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1. / 255)

# create the data generator for training...
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# ...and for validation
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# load the VGG16 network
# loading it without the fully connected layers at the top
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
for layer in base_model.layers: #adding the VGG16 layers
    model.add(layer)
model.add(Flatten(input_shape=model.output_shape[1:])) #add flatten layer
model.add(Dense(256, activation='relu')) #fully connected layers
model.add(Dropout(0.5)) #add dropout
model.add(Dense(1, activation='sigmoid')) #using sigmoid activation
print('Model loaded.')

# set the first 18 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# only training the fully connected layers
for layer in model.layers[:18]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# print the model summary for checking (eg. which layers are frozen)
model.summary()

#learning first phase
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#e arly stopping for the second phase
patience=10
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='smallships_weights.hdf5', save_best_only=True, verbose=1)
    
# allow the top of the convolution layers to learn
for layer in model.layers[14:18]:
    layer.trainable = True

# rerun the learning but this time with much lower learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

model.summary()

# second phase of learning
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=5000, # Because of the early stopping I set a large number
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpointer, early_stopping])
