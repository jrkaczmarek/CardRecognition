import cv2
import time
import preprocess
import constants
from helper import stackImages
from tensorflow import keras
import numpy as np

to_values = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'a', 10: 'j', 11: 'k', 12: 'q'}
to_colors = {0: 'club', 1: 'diamond', 2: 'heart', 3: 'spade'}

def capture():
    # loading neural network
    v_model = keras.models.load_model('./values_trained')
    c_model = keras.models.load_model('./colors_trained')

    # Settings
    cap = cv2.VideoCapture(constants.INTEGRATED_CAMERA)
    cap.set(3, constants.FRAME_WIDTH)
    cap.set(4, constants.FRAME_HEIGHT)
    cap.set(10, constants.BRIGHTNESS)

    flatten_card_set = []
    prev = 0

    while True:
        time_elapsed = time.time() - prev

        success, img = cap.read()

        if time_elapsed > 1. / constants.FRAME_RATE:
            prev = time.time()

            imgResult = img.copy()
            imgResult2 = img.copy()

            # preprocess the image
            thresh = preprocess.prepare(img)

            # find the set of corners that represent the cards
            four_corners_set = preprocess.find_corners(
                thresh, imgResult, draw=True)

            # warp the corners to form an image of the cards
            flatten_card_set = preprocess.flatten_card(
                imgResult2, four_corners_set)

            # get a crop of the borders for each of the cards
            cropped_images = preprocess.crop_left_corner(flatten_card_set)

            # isolate the value and color from the cards
            value_color_mapping = preprocess.split_value_and_color(
                cropped_images)

            # show the overall image
            cv2.imshow('Result', stackImages(0.85, [imgResult, thresh]))
            if value_color_mapping:
                value_im = value_color_mapping[0][0]/255
                color_im = value_color_mapping[0][1]/255

                value_im = cv2.resize(value_im, (140, 140))
                color_im = cv2.resize(color_im, (140, 140))

                value_im = value_im.reshape((1, 140, 140, 1))
                color_im = color_im.reshape((1, 140, 140, 1))

                value_result = v_model.predict(value_im)
                color_result = c_model.predict(color_im)

                value = to_values[np.argmax(value_result)]
                color = to_colors[np.argmax(color_result)]

                print("WYKRYTO")
                print('VALUE: ', value, 'COLOR:', color)

                cv2.imshow('Number', value_color_mapping[0][0])
                cv2.imshow('Color', value_color_mapping[0][1])

        wait = cv2.waitKey(1)
        if wait & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    capture()
