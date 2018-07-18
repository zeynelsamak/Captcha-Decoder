# extract_single_letters_from_captchas.py
import os
import os.path
import cv2
import glob
import imutils
import numpy as np
from matplotlib import pyplot as plt
from helpers import pre_process

CAPTCHA_IMAGE_FOLDER = "sgk_images"
OUTPUT_FOLDER = "extracted_letter_images7"
# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)

    captcha_correct_text = os.path.splitext(filename)[0]
    image, last, orig_img = pre_process(captcha_image_file)


    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(last.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        if len(contour) < 8:
            continue
        else:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
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
    # Save out each letter as a single image
    output = cv2.merge([image] * 3)

    if len(letter_image_regions) > 5:
        prev = None
        letter_image_regions2 = []
        for lir in letter_image_regions:

            if prev and lir[2] < 15:
                letter_image_regions2.append((prev[0], prev[1], prev[2] + lir[2], lir[3]))

            if lir[2] < 15:
                prev = lir
            else:
                prev = None
                letter_image_regions2.append(lir)
    else:
        letter_image_regions2 = letter_image_regions

    for letter_bounding_box, letter_text in zip(letter_image_regions2, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        # Get the folder to save the image in

        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{0}-{1}".format(str(count).zfill(6), filename))
        print (p)
        cv2.imwrite(p, letter_image)
        # increment the count for the current key
        counts[letter_text] = count + 1
    #     cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 244), 1)
    #
    # plt.imshow(output)
    # plt.show()