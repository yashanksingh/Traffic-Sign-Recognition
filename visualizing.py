import os
import cv2 as cv
import visualkeras
import matplotlib.pyplot as plot
from skimage import transform, exposure
from training import buildModel
from PIL import ImageFont
from collections import defaultdict
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


# VISUALIZING CLASSES AND SAMPLES
def visualize_classes(path, img_dir_path):
    rows = open(path).read().strip().split("\n")[1:]
    data = []
    for i, row in enumerate(rows):
        data.append([str(i), row.strip().split(',')[-1]])

    plot.rcParams["figure.figsize"] = [6.00, 9.00]
    plot.rcParams["figure.autolayout"] = True
    fig, axs = plot.subplots(1, 1)
    axs.axis('tight')
    axs.axis('off')
    axs.table(
        colLabels=['Class IDs', 'Class Names'],
        colWidths=[0.2, 0.8],
        cellText=data,
        cellLoc='left',
        loc='center'
    )
    plot.savefig("./Plots/classes-names.png")

    imgs = []
    n = len(os.listdir(img_dir_path))
    for i in range(n):
        image = cv.imread(img_dir_path + "/" + str(i) + ".png")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        imgs.append(image)
    fig = plot.figure(figsize=(10, 8))
    columns = 8
    rows = 6
    for i in range(n):
        fig.add_subplot(rows, columns, i+1)
        plot.imshow(imgs[i])
        plot.axis('off')
        plot.title(str(i))
    plot.suptitle("Samples from each Class")
    plot.savefig("./Plots/classes-samples.png")
    plot.show()


# SHOWING SAMPLES WITH BASIC PROCESSING APPLIED
def show_processed_samples(img_dir_path):
    images = []
    n = len(os.listdir(img_dir_path))
    for i in range(n):
        loaded_image = cv.imread(img_dir_path + "/" + str(i) + ".png")
        loaded_image = cv.cvtColor(loaded_image, cv.COLOR_BGR2GRAY)
        loaded_image = transform.resize(loaded_image, (32, 32))
        loaded_image = exposure.equalize_adapthist(loaded_image, clip_limit=0.1)
        images.append(loaded_image)
    fig = plot.figure(figsize=(10, 8))
    columns = 8
    rows = 6
    for i in range(n):
        fig.add_subplot(rows, columns, i + 1)
        plot.imshow(images[i], cmap='gray')
        plot.axis('off')
        plot.title(str(i))
    plot.suptitle("Processed Samples from each Class")
    plot.savefig("./Plots/classes-samples-preprocessed.png")
    plot.show()


# VISUALIZING NUMBER OF IMAGES IN EACH CLASS OF TRAINING SET WITH BAR GRAPH
def visualize_datasets(train_csvPath, validation_csvPath):
    class_ids = []
    images_per_class = []
    rows = open(train_csvPath).read().strip().split("\n")[1:]
    for row in rows:
        row = row.strip().split(",")
        class_ids.append(row[-2])
    num_classes = len(set(class_ids))
    for i in range(num_classes):
        images_per_class.append(class_ids.count(str(i)))

    plot.figure(figsize=(15, 8))
    plot.bar(range(num_classes), images_per_class)
    for index, value in enumerate(images_per_class):
        plot.text(index-0.25, value+40, str(value), rotation='vertical')
    plot.xticks(range(num_classes))
    plot.ylim(0, 2500)
    plot.title("Distribution of the Training Dataset")
    plot.xlabel("Class ID")
    plot.ylabel("Images")
    plot.savefig("./Plots/training-set-bar.png")
    plot.show()

    class_ids = []
    images_per_class = []
    rows = open(validation_csvPath).read().strip().split("\n")[1:]
    for row in rows:
        row = row.strip().split(",")
        class_ids.append(row[-2])
    num_classes = len(set(class_ids))
    for i in range(num_classes):
        images_per_class.append(class_ids.count(str(i)))

    plot.figure(figsize=(15, 8))
    plot.bar(range(num_classes), images_per_class)
    for index, value in enumerate(images_per_class):
        plot.text(index-0.25, value+20, str(value), rotation='vertical')
    plot.xticks(range(num_classes))
    plot.ylim(0, 1000)
    plot.title("Distribution of the Validation Dataset")
    plot.xlabel("Class ID")
    plot.ylabel("Images")
    plot.savefig("./Plots/validation-set-bar.png")
    plot.show()


# VISUALIZING ALL THE HIDDEN LAYERS IN THE NEURAL NETWORK
def show_hidden_layers(dimensions, classes_num):
    model = buildModel(dimensions, classes_num)

    color_map = defaultdict(dict)
    color_map[Conv2D]['fill'] = '#ffd166'
    color_map[Activation]['fill'] = '#ef476f'
    color_map[BatchNormalization]['fill'] = '#06d6a0'
    color_map[MaxPooling2D]['fill'] = '#118ab2'
    color_map[Flatten]['fill'] = '#073b4c'
    color_map[Dense]['fill'] = '#8338ec'
    color_map[Dropout]['fill'] = '#fb5607'

    font = ImageFont.truetype("arial.ttf", 14)
    visualkeras.layered_view(model, legend=True, color_map=color_map, font=font, to_file='./Plots/layered-view-neural-net.png').show()


# VISUALIZING FINAL GRAPHS FOR TRAINING AND VALIDATION LOSS AND ACCURACY AFTER TRAINING
def show_accuracy_loss_graphs(history, model_name):
    plot.figure(1)
    plot.plot(history.history['loss'])
    plot.plot(history.history['val_loss'])
    plot.legend(['Training', 'Validation'])
    plot.title('Loss')
    plot.xlabel('Epoch')
    plot.savefig(f"./Plots/{model_name}-training-validation-loss.png")

    plot.figure(2)
    plot.plot(history.history['accuracy'])
    plot.plot(history.history['val_accuracy'])
    plot.legend(['Training', 'Validation'])
    plot.title('Accuracy')
    plot.xlabel('Epoch')
    plot.savefig(f"./Plots/{model_name}-training-validation-accuracy.png")

    plot.show()


# FINAL PRE-TRAINING VISUALIZATION FUNCTION
def visualize(classes_csv, img_path, train_csv, valid_csv, dimensions):
    # SHOWING CLASSES, SAMPLES AND GRAPHS
    visualize_classes(classes_csv, img_path)
    show_processed_samples(img_path)
    visualize_datasets(train_csv, valid_csv)
    class_ids = []
    rows = open(train_csv).read().strip().split("\n")[1:]
    for row in rows:
        row = row.strip().split(",")
        class_ids.append(row[-2])
    classes_num = len(set(class_ids))
    show_hidden_layers(dimensions, classes_num)
