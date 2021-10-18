import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt


############################################################################################
############################### Pre-processing the dataset #################################
############################################################################################




####################################
### processing the training data ###
####################################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'dataset/mrlEyes_2018_01/training',
    target_size=(28, 28),
    batch_size=32,
    color_mode = "grayscale",
    class_mode='binary')

##################################
### preprocessing the test set ###
##################################
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/mrlEyes_2018_01/test',
    target_size=(28, 28),
    batch_size=32,
    color_mode = "grayscale",
    class_mode='binary')

############################################################################################
######################################## Model Building ####################################
############################################################################################ 


#########################
# inintialising the cnn #
#########################

cnn = Sequential()


####################
# building the cnn #
####################

cnn.add(Conv2D(filters=32, kernel_size=3,
               activation='relu', input_shape=(28, 28, 1)))
cnn.add(MaxPool2D(pool_size=2, strides=2))

####################
# Adding 2nd layer #
####################

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))


####################
# Adding 3rd layer #
####################

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

##############
# Flattening #
##############

cnn.add(Flatten())

###################
# full connection #
###################

cnn.add(Dense(units=128, activation='relu'))

################
# output layer #
################

cnn.add(Dense(units=1, activation='sigmoid'))

###########################################################################################
############################## Compiling and fitting the model ############################
###########################################################################################

cnn.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy'])
history = cnn.fit(x = training_set, validation_data=test_set, epochs=27)

#cnn.save('model_drowsiness_final_model.h5', overwrite=True)



# test_image = image.load_img('dataset/mrlEyes_2018_01/test/closed/s0005_00002_0_0_0_0_0_01.png', target_size = (28, 28))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = -1)
# result = cnn.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'closed'
# else:
#     prediction = "open"
# print(prediction)


############################################################################################
########################### Visualizing the model Accuracy and loss ########################
############################################################################################

##################
# model Accuracy #
##################

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')


##############
# model loss #
##############

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')
