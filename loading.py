import os
import numpy as np
import cv2 as cv
from skimage import transform, exposure
from time import time
from multiprocessing import Pool


# IMAGE PREPROCESSING FUNCTION
def image_processing(img):
    img = transform.resize(img, (32, 32))  # Resizing images to 32x32 pixels.
    img = exposure.equalize_adapthist(img, clip_limit=0.1)  # Applying Histogram Equalization to standardize lighting.
    img = img.astype("float32") / 255.0  # Normalizing image values between 0 and 1.
    return img


# IMAGE LOADING FUNCTION FOR MULTI-PROCESSING
def load_images(filename):
    loaded_image = cv.imread(filename)
    loaded_image = image_processing(loaded_image)
    return loaded_image


# LOADING TRAINING IMAGES FROM DATASET
def load_training(path):
    t1 = time()
    total_images = 0
    images = []
    labels = []
    dataset_root_dir_list = os.listdir(path)
    no_of_classes = len(dataset_root_dir_list)

    print("\nTotal Classes Detected:", no_of_classes)
    print("Importing Classes...")
    for x in range(no_of_classes):
        image_count = 0
        filenames = []
        dataset_class_dir_list = os.listdir(path + "/" + str(x))

        for y in dataset_class_dir_list:
            filenames.append(path + "/" + str(x) + "/" + y)
            labels.append(x)
            image_count += 1
        total_images += image_count

        t2 = time()
        with Pool(processes=4) as pool:
            result = pool.imap(load_images, filenames)
            for i in result:
                images.append(i)
        t2 = time() - t2

        print(f"Loaded Class {x}\t\tImages: {image_count:04d}\t\tTook {t2:07.4f} seconds\t\t{(t2*100)/image_count:.4f}ms/image\t\t{image_count/t2:.1f} images/s")
    t1 = time() - t1
    print(f"\nLoaded training dataset in {t1:.4f} seconds")
    print(f"Total Images Loaded: {total_images}")

    images = np.array(images)
    labels = np.array(labels)
    print()
    return images, labels


# LOADING VALIDATION IMAGES FROM DATASET
def load_validation(path, csvPath):
    t1 = time()
    total_images = 0
    images = []
    labels = []
    filenames = []
    rows = open(csvPath).read().strip().split("\n")[1:]

    print("\nLoading Validation Dataset...")
    for row in rows:
        (label, imagePath) = row.strip().split(",")[-2:]
        imagePath = imagePath[4:]
        imagePath = path + imagePath
        filenames.append(imagePath)
        labels.append(int(label))
        total_images += 1

    with Pool(processes=8) as pool:
        result = pool.imap(load_images, filenames)
        count = 0
        t2 = time()
        for i in result:
            if count > 0 and count % 1000 == 0:
                t2 = time() - t2
                print(f"Processed {count:05d} images in {t2:.4f} seconds\t\t{(t2*100)/1000:.4f}ms/image\t\t{1000/t2:.1f} images/s")
                t2 = time()
            images.append(i)
            count += 1

    t1 = time() - t1
    print(f"\nLoaded validation dataset in {t1:.4f} seconds")
    print(f"Total Images Loaded: {total_images}")

    images = np.array(images)
    labels = np.array(labels)
    print()
    return images, labels


# CHECKING DATA SHAPES FOR TRAINING AND VALIDATION SETS
def verify_shapes(train_images, train_labels, validation_images, validation_labels, dimensions):
    print("\nVerifying Data Shapes...")

    print(f"Training Set Shapes:\t\tImages: {train_images.shape}\t\tLabels: {train_labels.shape}")
    assert (train_images.shape[0] == train_labels.shape[0]), "The number of images is not equal to the number of labels in Training set"
    assert (train_images.shape[1:] == dimensions), "The dimensions of the Training images are wrong"

    print(f"Validation Set Shapes:\t\tImages: {validation_images.shape}\t\tLabels: {validation_labels.shape}")
    assert (validation_images.shape[0] == validation_labels.shape[0]), "The number of images is not equal to the number of labels in Validation set"
    assert (validation_images.shape[1:] == dimensions), "The dimensions of the Validation images are wrong"
    print()


# FINAL DATASET LOADING FUNCTION
def load_dataset(train_path, valid_path, valid_csv, dimensions, preloaded=False):
    if not preloaded:
        # PREPARING ARRAY OF IMAGES WITH CORRESPONDING CLASS IDS
        X_train, Y_train = load_training(train_path)
        X_validation, Y_validation = load_validation(valid_path, valid_csv)

        # CHECKING IF DATASET WAS LOADED PROPERLY
        verify_shapes(X_train, Y_train, X_validation, Y_validation, dimensions)

        np.save("./Datasets/GTSRB/X_train.npy", X_train)
        np.save("./Datasets/GTSRB/Y_train.npy", Y_train)
        np.save("./Datasets/GTSRB/X_validation.npy", X_validation)
        np.save("./Datasets/GTSRB/Y_validation.npy", Y_validation)
    else:
        print("\nLoading dataset from preloaded numpy arrays...")
        t1 = time()
        X_train = np.load("./Datasets/GTSRB/X_train.npy")
        Y_train = np.load("./Datasets/GTSRB/Y_train.npy")
        t1 = time() - t1
        print(f"Loaded training dataset in {t1:.4f} seconds")
        t2 = time()
        X_validation = np.load("./Datasets/GTSRB/X_validation.npy")
        Y_validation = np.load("./Datasets/GTSRB/Y_validation.npy")
        t2 = time() - t2
        print(f"Loaded validation dataset in {t2:.4f} seconds")

        # CHECKING IF DATASET WAS LOADED PROPERLY
        verify_shapes(X_train, Y_train, X_validation, Y_validation, dimensions)

    return X_train, Y_train, X_validation, Y_validation
