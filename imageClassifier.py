''' From tutorial https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html'''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2002
nb_validation_samples = 802
epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3)

# model with 2 fully connected layers
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# the model now oytputs 3D feature maps (height, width, features)
# model output with a single unit and sigmoid activation (useful for binacy classification)
model.add(Flatten()) # this converts the 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# to train our model	
model.compile(loss='binary_crossentropy', 
			optimizer='rmsprop', 
			metrics=['accuracy'])

# Prepare the data
batch_size = 16

# data augmentation configuration used for training
train_datagen = ImageDataGenerator(rescale=1./255, 
									shear_range=0.2, 
									zoom_range=0.2, 
									horizontal_flip=True)

# data augmentation configuration used for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generator that will read pictures from training data and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(train_data_dir, # target directory
													target_size=(img_width, img_height), # all images will be resized to 150x150
													batch_size=batch_size,
													class_mode='binary') # since we use binart_crossentropy loss, we need binary labels

# Similar generator but for validation data
validation_generator = test_datagen.flow_from_directory(validation_data_dir,
														target_size=(img_width, img_height),
														batch_size=batch_size,
														class_mode='binary')

# Use this generator to train our model
model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)
model.save_weights('first_try.h5') # save the weights 
