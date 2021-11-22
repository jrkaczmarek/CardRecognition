import cv2
import numpy as np


def contrast(img, value):
    brightness = 30
    shadow = brightness
    highlight = 255

    # add the brightness
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow
    img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

    # add the constrast
    f = 131 * (value + 127) / (127 * (131 - value))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

    return img


def fill(image, size, color=[255, 255, 255]):
    """Add a border to image to fill the given size*size square.

    Parameters:
    argument1 (image): Image
    argument2 (int): The size of the square you need
    argument3 (optional) (list): RGB color of a border

    Returns:
    list: Images of value and color of card

    """
    l = int((size - image.shape[1]) / 2) 
    h = int((size - image.shape[0]) / 2) 

    border = cv2.copyMakeBorder(
        image,
        top=h,
        bottom=h,
        left=l,
        right=l,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

    resized = cv2.resize(border, (size, size), interpolation = cv2.INTER_AREA)

    return resized
