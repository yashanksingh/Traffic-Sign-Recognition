import numpy as np
import cv2 as cv
from keras import models

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 640
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv.FONT_HERSHEY_SIMPLEX

# SET UP THE VIDEO CAMERA
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
# pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
# model = pickle.load(pickle_in)
model = models.load_model('sign_classifier_v1.4.model')


def preprocessing(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.equalizeHist(image)
    image = image / 255
    return image


def getClassName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vehicles'
    elif classNo == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vehicles over 3.5 metric tons'


def predictImage(imagePath):
    image = cv.imread(imagePath)
    image = cv.resize(image, (32, 32))
    image = preprocessing(image)
    image = image.reshape(1, 32, 32, 1)
    prediction = model.predict(image)
    index = np.argmax(prediction)
    probability = np.amax(prediction)
    print(index)
    print(getClassName(index))
    print(probability)
    print(prediction)


while True:
    # READ IMAGE
    success, imgOriginal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    # SHOW RESULTS
    if probabilityValue > threshold:
        cv.putText(imgOriginal, "CLASS: " + str(classIndex) + " " + str(getClassName(classIndex)), (20, 35),
                   font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75),
                   font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("Test", imgOriginal)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
