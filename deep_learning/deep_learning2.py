# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow


def save():
    classifier.save("saved_model/gestures.h5")
    print(classifier.summary())


def create_model():
    classifier = Sequential()  # Initialising the CNN
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def train_data(classifier):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/rps_train',
                                                     target_size=(64, 64), batch_size=2, class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/rps_test',
                                                target_size=(64, 64), batch_size=2, class_mode='binary')

    classifier.fit_generator(training_set, steps_per_epoch=1000,
                             epochs=5, validation_data=test_set, validation_steps=1000)

    save()


if __name__ == "__main__":

    if os.path.exists("saved_model/gestures.h5"):
        classifier = load_model("saved_model/gestures.h5")
        print(classifier.summary())
        print("model found")
    else:
        classifier = create_model()
        train_data(classifier)
