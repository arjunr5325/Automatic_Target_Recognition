import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten, Dropout, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import imutils
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from IPython.display import Image, display


BASE_OUTPUT = "/content/drive/MyDrive/Output_dir"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# Constants
INIT_LR = 1e-4
NUM_EPOCHS = 11
BATCH_SIZE = 64

# Specify the dataset directory structure
dataset_dir = "/content/drive/MyDrive/data_YOLO"
class_names = ["military_ships", "military_tanks"]

# Load data from CSV files and organize images
data = []
labels = []
bboxes = []
imagePaths = []

for class_name in class_names:
    annotations_path = os.path.join(dataset_dir, "annotations", f"{class_name}.csv")
    images_dir = os.path.join(dataset_dir, "images", class_name)

    df = pd.read_csv(annotations_path, sep=";", header=None)  

    for index, row in df.iterrows():
        image_filename = row.iloc[0]
        image_path = os.path.join(images_dir, image_filename)
        print("Loading image:", image_path)

        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image:", image_path)
        else:
            y_ = image.shape[0]
            x_ = image.shape[1]
            targetSize = 224
            x_scale = targetSize / float(x_)
            y_scale = targetSize / float(y_)
            image = cv2.resize(image, (targetSize, targetSize))

            x = int(np.round(float(row.iloc[4]) * x_scale))
            y = int(np.round(float(row.iloc[5]) * y_scale))
            xmax = int(np.round(float(row.iloc[6]) * x_scale))
            ymax = int(np.round(float(row.iloc[7]) * y_scale))

            data.append(np.array(image))
            bboxes.append((x, y, xmax, ymax))
            imagePaths.append(image_path)
            labels.append(row.iloc[3])

data = np.array(data, dtype="float32") / 255
bboxes = np.array(bboxes, dtype="float32") / 255
imagePaths = np.array(imagePaths)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

# Partition the data into training and testing splits
split = train_test_split(data, labels, bboxes, imagePaths,
	test_size=0.20, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# Write the testing image paths to disk
test_paths_file = "/content/drive/MyDrive/test_path.txt"
print("[INFO] saving testing image paths...")
with open(test_paths_file, "w") as f:
    f.write("\n".join(testPaths))

# Load the VGG16 network, ensuring the head FC layers are left off
num_classes = 2  # Change this to the number of classes in your task
INIT_LR = 1e-4
vgg = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
# Freeze all VGG layers so they will not be updated during the training process
vgg.trainable = False
# Flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# Construct a fully-connected layer header to output the predicted bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
# Construct a second fully-connected layer head, this one to predict the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(num_classes, activation="softmax", name="class_label")(softmaxHead)
# Put together our model which accepts an input image and then outputs
# bounding box coordinates and a class label
model = Model(
    inputs=vgg.input,
    outputs=(bboxHead, softmaxHead))
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}
# Define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}
# Initialize the optimizer, compile the model, and show the model summary
opt = Adam(lr=INIT_LR)  # Use the defined INIT_LR
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

# Construct dictionaries for our target training and testing outputs
trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}
testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

# Train the network for bounding box regression and class label prediction
print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1)    

# Serialize the label binarizer to disk using pickle
print("[INFO] saving label binarizer...")
with open(LB_PATH, 'wb') as lb_file:
    pickle.dump(lb, lb_file)

# Plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# Loop over the loss names
for (i, l) in enumerate(lossNames):
    # Plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

# Save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("losses.png")
plt.close()

# Create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
    label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
    label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# Save the accuracies plot
plt.savefig("accuracies.png")

# Define the path to your input image
args = {"input": "/content/drive/MyDrive/teststuff/test1.jpg"}  # Replace with your image path

# Define the model and label binarizer paths
MODEL_PATH = "/content/drive/MyDrive/Output_dir/detector.h5"  # Replace with your actual model path
LB_PATH = "/content/drive/MyDrive/Output_dir/lb.pickle"  # Replace with your actual label binarizer path

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model(MODEL_PATH)
lb = pickle.loads(open(LB_PATH, "rb").read())

# load the input image (in Keras format) from disk and preprocess
# it, scaling the pixel intensities to the range [0, 1]
image = load_img(args["input"], target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# predict the bounding box of the object along with the class
# label
(boxPreds, labelPreds) = model.predict(image)
(startX, startY, endX, endY) = boxPreds[0]

# determine the class label with the largest predicted
# probability
i = np.argmax(labelPreds, axis=1)
label = lb.classes_[i][0]

# Check if the label is either "tank" or "warship" and if the highest confidence score is below the threshold
confidence_threshold = 0.5  # Adjust this threshold as needed
highest_confidence = labelPreds[0][i]
if (label == "tank" or label == "warship") and highest_confidence >= confidence_threshold:
    image = cv2.imread(args["input"])
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 255, 0), 2)

output_image_path = "output_image.jpg"  # You can save the modified image to a file first
cv2.imwrite(output_image_path, image)
display(Image(filename=output_image_path))