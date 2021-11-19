import cv2
import numpy as np

from helper import contrast, stackImages, fill


def prepare(image):
    """Applies canny filter and dilation on the given image.

    Parameters:
    argument1 (image): Image of a card

    Returns:
    image: Processed image

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((3, 3))
    dilation = cv2.dilate(canny, kernel=kernel, iterations=2)

    return dilation


def find_corners(processed, original, draw=False):
    """Finds the card corners, if draw=True draws rectangle where the card is 
    and circles where the corners are placed on the original image.

    Parameters:
    argument1 (image): Processed image of a card
    argument2 (image): Original image of card

    Returns:
    list: Card corners

    """

    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    corners = []

    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if area > 10000:
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, closed=True)
            numCorners = len(approx)

            if numCorners == 4:
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # Sort by x
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # SortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # Now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(
                    sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                corners.append(finalOrder)

                if draw:
                    for a in approx:
                        cv2.circle(
                            original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)

    return corners


def flatten_card(image, corners):
    """Flatten the card and rotate it.

    Parameters:
    argument1 (image): Original image of card
    argument2 (list): List of corners

    Returns:
    list: Images of flattened and rotated properly cards

    """

    width, height = 200, 300
    image_outputs = []

    for corner in corners:
        # Get the 4 corners of the card
        pts1 = np.float32(corner)

        # Define which corners
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        output = cv2.warpPerspective(image, matrix, (width, height))
        image_outputs.append(output)

    return image_outputs


def crop_left_corner(flattened_images):
    """Crop left corner to see the value and color.

    Parameters:
    argument1 (list): Flattened images of cards

    Returns:
    list: Images of left card corner

    """

    corner_images = []
    for img in flattened_images:
        crop = img[5:120, 5:32]  # Depends on image resolution
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # Threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        canny = cv2.Canny(bilateral, 24, 40)
        kernel = np.ones((3, 3))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        # Append the thresholded image and the original one
        corner_images.append([result, gray])

    return corner_images


def split_value_and_color(cropped_images):
    """Split the value and color of the card.

    Parameters:
    argument1 (list): Cropped images of cards

    Returns:
    list: Images of value and color of card

    """
    value_color_mapping = []

    for img, original in cropped_images:
        contours, hier = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the largest two contours
        highest_two = dict()
        for contour in contours:
            area = cv2.contourArea(contour)

            # Fill smaller rectangles with black
            if area < 2000:
                cv2.fillPoly(img, pts=[contour], color=0)
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            # append the contour and the perimeter
            highest_two[area] = [contour, perimeter]

        # Select the largest two
        mapping = []

        for area in sorted(highest_two)[0:2]:
            contour = highest_two[area][0]
            perimeter = highest_two[area][1]
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)
            x, y, w, h = cv2.boundingRect(approx)
            crop = original[y:y + h][:]

            sharpened = contrast(crop, 50)

            for i in range(sharpened.shape[0]):
                for j in range(sharpened.shape[1]):
                    if sharpened[i, j] < 220:
                        sharpened[i, j] = max(0, sharpened[i, j] - 100)
                    if sharpened[i, j] > 221:
                        sharpened[i, j] = 255

            mapping.append([sharpened, y])

        # First store value and then color
        mapping.sort(key=lambda x: x[1])

        # Append results
        if len(mapping) == 2:
            value_color_mapping.append([mapping[0][0], mapping[1][0]])

    return value_color_mapping


def get_data(image, size=140):
    """Get the important data from card image.

    Parameters:
    argument1 (image): Image of a card

    Returns:
    tuple: Image of a value and image of a color

    """
    p = prepare(image)
    f = find_corners(p, image, draw=True)
    fl = flatten_card(image, f)
    c = crop_left_corner(fl)
    final = split_value_and_color(c)
    value, color = fill(final[0][0], size), fill(final[0][1], size)

    return value, color


# TESTING
if __name__ == '__main__':
    img = cv2.imread('sample.jpg')
    value, color = get_data(img)

    cv2.imshow("value", value)
    cv2.imshow("color", color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
