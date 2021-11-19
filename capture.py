import cv2
import time
import preprocess
import constants

from helper import stackImages


def capture():
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
                cv2.imshow('Number', value_color_mapping[0][0])
                cv2.imshow('Color', value_color_mapping[0][1])

        wait = cv2.waitKey(1)
        if wait & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    capture()
