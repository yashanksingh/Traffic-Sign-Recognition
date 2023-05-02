from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import pandas as pd


def buildModel(dimensions, classes_num):
    model = Sequential()
    chanDim = -1

    # layer: Conv -> RELU -> BN -> POOL
    model.add(Conv2D(32, (4, 4), input_shape=dimensions, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # first set of (CONV => RELU => BN) * 2 => POOL
    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second set of (CONV => RELU => BN) * 3 => POOL
    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    # FC layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax
    model.add(Dense(classes_num))
    model.add(Activation("softmax"))

    return model


def train(dimensions, X_train, Y_train, X_validation, Y_validation, epochs, batch_size, model_name):
    # ONE-HOT ENCODING OF LABELS
    classes_num = len(set(Y_train))
    Y_train = to_categorical(Y_train, classes_num)
    Y_validation = to_categorical(Y_validation, classes_num)

    # AUGMENTATION OF IMAGES: TO MAKEIT MORE GENERIC
    dataGen = ImageDataGenerator(width_shift_range=0.1,  # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFERS TO NO. OF  PIXELS EG 10 PIXELS
                                 height_shift_range=0.1,
                                 zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                                 shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                                 rotation_range=10)  # DEGREES
    dataGen.fit(X_train)

    # IMPLEMENTING OPTIMIZER AND COMPILING MODEL
    opt = Adam(learning_rate=0.001, decay=0.001 / (epochs * 0.5))
    model = buildModel(dimensions, classes_num)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # CHECKING MODEL LAYERS
    model.summary()

    # MODEL TRAINING
    history = model.fit(
        dataGen.flow(X_train, Y_train, batch_size=batch_size),
        validation_data=(X_validation, Y_validation),
        epochs=epochs,
        steps_per_epoch=X_train.shape[0] // batch_size,
        shuffle=True
    )

    # SAVING THE MODEL
    pd.DataFrame.from_dict(history.history).to_csv('./Models/' + model_name[:-5] + 'csv', index=False)
    model.save('./Models/' + model_name)
    model.save('./Models/' + model_name[:-5] + 'h5')

    return history
