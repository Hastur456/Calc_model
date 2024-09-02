import cv2 as cv
import os


def delete_alfa_channel(path):
    path_ = path
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    trans_mask = image[:, :, 3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    os.remove(path_)
    cv.imwrite(path_, image)


def get_all_paths(dir_):
    list_catalog = []
    list_with_all_images = []

    if os.path.isdir(dir_):
        main_list_dir = [os.path.join(dir_, path) for path in os.listdir(dir_)]
        for i in range(len(main_list_dir)):
            list_dir = [os.path.join(main_list_dir[i], path) for path in os.listdir(main_list_dir[i])]
            list_catalog.append(list_dir)
        for j in range(len(list_catalog[0])):
            list_dir = [os.path.join(list_catalog[0][j], path) for path in os.listdir(list_catalog[0][j])]
            list_with_all_images.append(list_dir)

    answer_list = []
    for lst in list_with_all_images:
        for el in lst:
            answer_list.append(el)

    return answer_list


def image_convert(img):
    img = cv.imread(img, 0)
    img = cv.cvtColor(img, cv.COLOR_BAYER_RG2GRAY)
    return img


def find_conturs(img):
    img = image_convert(img)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    T, thresh_img = cv.threshold(blurred, 215, 255, cv.THRESH_BINARY)
    thresh_img = 255 - thresh_img
    conts, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return conts


def changing_words_images(dir_):
    conturs = find_conturs(dir_)
    img = image_convert(dir_)

    x, y, w, h = cv.boundingRect(conturs[0])
    word_image = img[y:y + h, x:x + w]
    word_image = cv.cvtColor(word_image, cv.COLOR_BAYER_GR2GRAY)
    word_image = cv.resize(word_image, (28, 28))

    os.remove(dir_)
    cv.imwrite(dir_, word_image)


def resize_dataset(dir_, kernel_size=(64, 64)):
    img = image_convert(dir_)
    img = cv.resize(img, kernel_size)

    os.remove(dir_)
    cv.imwrite(dir_, img)


def changing_dataset(func, dirs_images):
    for image in dirs_images:
        func(image)


_dir_ = "Dataset/Train/5/58be7fcf6965d.png"
main_dir = "Dataset"
