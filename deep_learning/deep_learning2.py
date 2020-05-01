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
    classifier.save("/content/drive/My Drive/FYP Colab/gestures.h5")
    print(classifier.summary())


def create_model():
    classifier = Sequential()  # Initialising the CNN

    # POSSIBLE FIRST LAYER? COULD BE OVERKILL/ OR COULD EVEN OVERFIT
    # classifier.add(Conv2D(256, (2, 2), input_shape=(64, 64, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 1
    classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=128,
                         activation='relu'))  # """UNCOMMENT THIS IS ACCURACY DOESNT GO UP | MAYBE MAKE IT 128 INITIALLY"""
    classifier.add(Dropout(0.5))
    # classifier.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def create_model2():
    classifier = Sequential()
    classifier.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3)))
    classifier.add(MaxPooling2D(pool_size=3))
    classifier.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=3))
    classifier.add(Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=3))
    classifier.add(Flatten())
    classifier.add(Dense(3, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


def train_data(classifier):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('/content/drive/My Drive/FYP Colab/rockpaperscissors/training_set',
                                                     target_size=(100, 100),
                                                     batch_size=1, class_mode='categorical')

    test_set = test_datagen.flow_from_directory('/content/drive/My Drive/FYP Colab/rockpaperscissors/test_set',
                                                target_size=(100, 100),
                                                batch_size=1, class_mode='categorical')

    classifier.fit_generator(training_set, steps_per_epoch=2492,
                             epochs=25,
                             validation_data=test_set,
                             validation_steps=2492)

    save()


def train_data2(classifier):
    hist = classifier.fit('/content/drive/My Drive/FYP Colab/rockpaperscissors/training_set', validation_split=0.2,
                          epochs=2, batch_size=1)
    score = classifier.evaluate('/content/drive/My Drive/FYP Colab/rockpaperscissors/test_set', verbose=0)
    print(score[1])


if __name__ == "__main__":

    if os.path.exists("/content/drive/My Drive/FYP Colab/gestures.h5"):
        classifier = load_model("/content/drive/My Drive/FYP Colab/gestures.h5")
        print(classifier.summary())
        print("model found")
    else:
        classifier = create_model2()
        train_data(classifier)