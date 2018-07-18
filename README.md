# Capthca-Decoder
A captcha solver that using machine learning(Nearest Neigbour) and deep learning(CNN,keras)

# Decoding Steps

   * Data Collection
       - Collecting the captcha images with the ground truth text
   * Preprocessing
       - Extract the letter from the captcha images by using **opencv** methods *(threshold > erode/dilate > countour)*
       - Save the each letter to related the file - *For example: All 0s to the the 0, all 9 letters to the 9 file*
   * Training
       - Build the CNN network by using keras or use scikit-learn to model NNeigbour classifier
       - Train the model with the extracted letters
       - When the model performance reach the optimum level, save the model weights for inference
   * Testing
       - Use local file, url or base64 based captcha image
       - Extract the letter as in *preprocessing* stage
       - Run the inference model on each letter that extracted
       - Combine the letter prediction in order, and return the final result


## Result


![alt text](https://i.imgur.com/RuU99I2.jpg "Example Result")

## TODO
  - Extend the explaination of the work
  - Train the network without to parsing the letter. Feed whole image into the netwokr and predict the captcha text

## Acknowledgement

This work was inspired by work of [Adam Geitgey's medium post](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710) 




      
