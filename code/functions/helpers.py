# import the necessary packages
from PIL import Image
import pickle
import cv2
import numpy as np
import imutils
from keras.models import load_model
import base64
import io
from sklearn.neighbors import KNeighborsClassifier
import urllib
import certifi
from keras import backend as K

# MODEL_FILENAME = "model/captcha_model2.hdf5"
# MODEL_FILENAME2 = 'model/NN_model.sav'
MODEL_FILENAME = "model/captcha_model_final.hdf5"
MODEL_FILENAME2 = 'model/NN_model2.sav'

MODEL_LABELS_FILENAME = "model/model_labels2.dat"

def predict_captcha_text(image_file,img_type, mod='keras'):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # Load the trained neural network
    if mod=='keras':
        model = load_model(MODEL_FILENAME)
    else:
        print('scikit')
        model = pickle.load(open(MODEL_FILENAME2, 'rb'))


    image, last, orig_img = pre_process(image_file,img_type)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(last.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        if len(contour) < 9:
            continue
        else:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
            #         print (w,h,w/h)
            if w / h > 2:
                triple_width = int(w / 3)
                letter_image_regions.append((x, y, triple_width, h))
                letter_image_regions.append((x + triple_width, y, triple_width, h))
                letter_image_regions.append((x + 2 * triple_width, y, triple_width, h))

            elif w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    if len(letter_image_regions) > 5:
        prev = None
        letter_image_regions2 = []
        for lir in letter_image_regions:

            if prev:
                if lir[2] < 15:
                    letter_image_regions2.append((prev[0], prev[1], prev[2] + lir[2], 25))
                if lir[2] > 15:
                    letter_image_regions2.append(prev)
                    letter_image_regions2.append(lir)
                prev = None
            else:
                if lir[2] < 15:
                    prev = lir
                else:
                    prev = None
                    letter_image_regions2.append(lir)
    else:
        letter_image_regions2 = letter_image_regions

    # loop over the lektters
    for i, letter_bounding_box in enumerate(letter_image_regions2):
        # Grab the coordinates of the letter in the image
        if i > 4:
            break
        x, y, w, h = letter_bounding_box
        if x < 2:
            x = 2
        if y < 2:
            y = 2
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        #         print('shape',letter_image.shape)
        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        if mod=='keras':
            prediction = model.predict(letter_image)
        else:

            prediction = model.predict(letter_image.reshape(1, 400) / 255.0)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)
    K.clear_session()
    captcha_text = "".join(predictions)
#    cv2.imwrite('static/uploads/'+captcha_text+'.jpg',orig_img)
    return captcha_text


def base64toImg(b64_string):

    # reconstruct image as an numpy array
    img = io.BytesIO(base64.b64decode(b64_string))
    img = Image.open(img)
    # finally convert RGB image to BGR for opencv
    # and save result
    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) #np.array(img)

    return cv2_img


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image




def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    resp = urllib.request.urlopen(url, cafile=certifi.where())#urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

def pre_process(image_file,img_type):
    # image1 = cv2.imread(image_file)
    # image1 = url_to_image(image_file)
    if img_type=='file':
        image1 = cv2.imread(image_file)
    elif img_type=='url':
        image1 = url_to_image(image_file)
    else:
        image1 = base64toImg(image_file)


    # load the example image and convert it to grayscale
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image[0:10, :] = 255
    image[40:, :] = 255
    image[:, 136:] = 255

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)

    new = thresh.copy()
    new[new == 255] = 1
    hist = np.sum(new, axis=0)
    hist[hist < 7] = 0
    k = np.where(hist == 0)
    last2 = thresh.copy()
    last2[:, k] = 0


    s = 255 - cv2.erode(last2, kernel, iterations=1)
    s = cv2.medianBlur(s, 3)

    return s, last2, image1