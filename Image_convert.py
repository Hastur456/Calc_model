import cv2 as cv
import keras
import numpy as np
import matplotlib.pyplot as plt


def image_convert(img):
    img = cv.imread(img, 0)
    img = cv.cvtColor(img, cv.COLOR_BAYER_BG2GRAY)
    return img


def find_conturs(img):
    img = image_convert(img)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    t, thresh_img = cv.threshold(blurred, 215, 255, cv.THRESH_BINARY)
    thresh_img = 255 - thresh_img
    conts, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return conts


def get_letters_images(img):
    conturs = find_conturs(img)
    img = image_convert(img)

    letter_images = []
    for idx, contur in enumerate(conturs):
        x, y, w, h = cv.boundingRect(contur)
        letter_image = img[y:y + h, x:x + w]
        letter_images.append((x, w, letter_image))

    letter_images.sort(key=lambda n: n[0])
    return letter_images


def show_symbols(img):
    fig, ax = plt.subplots(10, 3, figsize=(5, 10))
    fig.subplots_adjust()
    for axi, image in zip(ax.flat, get_letters_images(img)):
        axi.imshow(image[2], cmap='rgb')
    fig.show()
    fig.waitforbuttonpress()


def predict_classification_model(model: any, letter_image, classes: list, im_size=45):
    image_ = cv.copyMakeBorder(letter_image, 4, 4, 4, 4,
                               borderType=cv.BORDER_CONSTANT, value=[255])
    image_ = cv.resize(image_, (im_size, im_size), interpolation=cv.INTER_AREA).astype("uint8")
    image_ = np.expand_dims(image_, axis=0)

    predict_ = model.predict(image_/255.)
    result = np.argmax(predict_, axis=1)
    return classes[result[0]]


def image_to_string(img, model: any, classes: list):
    letters = get_letters_images(img)

    string_out = ""
    for i in range(len(letters)):
        string_out += predict_classification_model(model, np.array(letters[i][2]), classes)
    return string_out


img_ = "Test_Images/5pls3_2.jpg"

model_ = keras.models.load_model("Models\\Model14.keras")
classes_ = ['(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '-', '+']

answer = image_to_string(img_, model_, classes_)
print(answer)
print(f"Equals: {eval(answer)}")
