from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow


def save():
    classifier.save("gestures.h5")
    print(classifier.summary())


def create_model2():  # model creation
    classifier = Sequential() # adding layers to average out pixel calues and condense amount of pixels
    classifier.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(100, 100, 3)))
    classifier.add(MaxPooling2D(pool_size=4))
    classifier.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=4))
    classifier.add(Flatten()) # flattening image into an array
    classifier.add(Dense(3, activation='softmax')) # categorising pixels
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # compiling result
    return classifier


def train_data(classifier):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # creating the parameters for how a training image will be analysed
    training_set = train_datagen.flow_from_directory('dataset/rockpaperscissors/training_set',
                                                     target_size=(100, 100),
                                                     batch_size=1, class_mode='categorical')

    # creating the parameters for how a test image will be analysed
    test_set = test_datagen.flow_from_directory('dataset/rockpaperscissors/test_set',
                                                target_size=(100, 100),
                                                batch_size=1, class_mode='categorical')

    # training the model
    classifier.fit_generator(training_set, steps_per_epoch=3086,
                             epochs=20,
                             validation_data=test_set,
                             validation_steps=3086)

    save() # saving model


if __name__ == "__main__":

    if os.path.exists("gestures.h5"): # if a model already exists load it
        classifier = load_model("gestures.h5")
        print(classifier.summary())
        print("model found")
    else: # if not train a new model
        classifier = create_model2()
        train_data(classifier)