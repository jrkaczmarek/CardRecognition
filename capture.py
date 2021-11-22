from numpy.core.fromnumeric import trace
import cv2
import time
import preprocess
import constants
import numpy as np

from helper import stackImages
from tensorflow import keras


def recognize(value_color_mapping, v_model, c_model, show=False):
    pred = []
    for i in range(len(value_color_mapping)):
        # Gray scale
        value_im = value_color_mapping[i][0] / 255
        color_im = value_color_mapping[i][1] / 255

        # Resize to 140x140
        value_im = cv2.resize(value_im, (140, 140))
        color_im = cv2.resize(color_im, (140, 140))

        # Reshape
        value_im = value_im.reshape((1, 140, 140, 1))
        color_im = color_im.reshape((1, 140, 140, 1))

        # Predict using value and color model
        value_result = v_model.predict(value_im)
        color_result = c_model.predict(color_im)

        # Translate 
        value = constants.TO_VALUES[np.argmax(value_result)]
        color = constants.TO_COLORS[np.argmax(color_result)]

        pred.append(f'VALUE: {value}\nCOLOR: {color}')

    return pred


def show_text(pred, corners, img):
    for i in range(0, len(pred)):
        # figure out where to place the text
        c = np.array(corners[i])
        corners_flat = c.reshape(-1, c.shape[-1])
        startX = corners_flat[0][0] + 0
        halfY = corners_flat[0][1] - 50
        pred_list = pred[i].split('\n')

        font = cv2.FONT_HERSHEY_COMPLEX
        gap = 0
        # show the text
        for j in pred_list:
            cv2.putText(img, j, (startX, halfY + gap), font, 0.8, (50, 205, 50), 2, cv2.LINE_AA)
            gap += 30


def capture():
    # loading neural network
    v_model = keras.models.load_model('./values_trained')
    c_model = keras.models.load_model('./colors_trained')

    # Settings
    cap = cv2.VideoCapture(0)
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
            if value_color_mapping:
                prediction = recognize(value_color_mapping, v_model, c_model)
                show_text(prediction, four_corners_set, imgResult)

            cv2.imshow('Result', imgResult)

        wait = cv2.waitKey(1)
        if wait & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    capture()
