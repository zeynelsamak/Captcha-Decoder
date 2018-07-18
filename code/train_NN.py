#train_model.py
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from helpers import resize_to_fit
LETTER_IMAGES_FOLDER = "extracted_letter_images4"
MODEL_FILENAME = "captcha_model4.hdf5"
MODEL_LABELS_FILENAME = "model_labels4.dat"
# initialize the data and labels
data = []
labels = []
# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
#     print (image_file)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)
    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)
    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]
    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)


x_tr = X_train.reshape(X_train.shape[0],400)
x_test = X_test.reshape(X_test.shape[0],400)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_tr, Y_train)

print ('model score:',neigh.score(x_test,Y_test))

NN_filename = 'NN_model.sav'
pickle.dump(neigh, open(NN_filename, 'wb'))

#for prediction
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)
# load the model from disk
loaded_model = pickle.load(open(NN_filename, 'rb'))
result = loaded_model.score(x_test, Y_test)
print(result)