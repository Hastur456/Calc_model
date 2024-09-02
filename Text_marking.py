import cv2 as cv


def image_convert(image):
    img = cv.imread(image, 0)
    img = cv.cvtColor(img, cv.COLOR_BAYER_RG2GRAY)
    return img


def find_conturs(img):
    img = image_convert(img)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    T, thresh_img = cv.threshold(blurred, 215, 255, cv.THRESH_BINARY)
    thresh_img = 255 - thresh_img
    conts, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return conts


def find_coordinates_symbols(conturs):
    card_coordinates = []
    for i in range(len(conturs)):
        x, y, w, h = cv.boundingRect(conturs[i])
        card_coordinates.append((x, y, x + w, y + h))
    return card_coordinates


def draw_rectangle_around_symbol(coordinates, img):
    img = image_convert(img)
    for coordinate in coordinates:
        cv.rectangle(img, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 0, 0), 2)
    cv.imshow("Image", img)
    cv.waitKey(0)


image_ = "Test_Images\\5pls3.png"

conturs_ = find_conturs(image_)
coordinates_ = find_coordinates_symbols(conturs_)
draw_rectangle_around_symbol(coordinates_, image_)
