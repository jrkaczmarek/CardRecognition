from tensorflow import keras
from preprocess import *
from capture import recognize, show_text

if __name__ == '__main__':
    # Load models
    v_model = keras.models.load_model('./values_trained')
    c_model = keras.models.load_model('./colors_trained')

    for i in range(22, 0, -1):
        name = f'test/{i}.jpg'
        img = cv2.imread(name)
        img_res = img.copy()
        p = prepare(img)
        # cv2.imshow("Prepared", p)
        f = find_corners(p, img_res, draw=True)
        # cv2.imshow("Rectangle and corners", img)
        fl = flatten_card(img, f)

        c = crop_left_corner(fl)

        final = split_value_and_color(c)

        if final:
            value, color = fill(final[0][0], 180), fill(final[0][1], 180)
            prediction = recognize(final, v_model, c_model)
            show_text(prediction, f, img_res)

        cv2.imshow("value", value)
        cv2.imshow("color", color)
        cv2.imshow("Original", img_res)

        cv2.waitKey(0)
        cv2.destroyAllWindows()