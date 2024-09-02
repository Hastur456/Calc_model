import numpy as np
import cv2
import Classification_model
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.utils import to_categorical

x = []
y = []
datadir = "C:\extracted_images"
for folder in os.listdir(datadir):
    path = os.path.join(datadir, folder)
    for images in os.listdir(path)[:400]:
        img = cv2.imread(os.path.join(path, images))
        x.append(img)
        y.append(folder)

print(len(x))
print(len(y))
print(f'labels : {list(set(y))}')

X = []
for i in range(len(x)):
    img = x[i]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_image = 255 - cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    threshold_image = cv2.resize(threshold_image, (45, 45))
    X.append(threshold_image)

print(len(X))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_train = X_train/255.
X_test = X_test/255.

print(X_train.shape)
print(X_test.shape)
print(Y_train)
print(Y_test.shape)


model = Classification_model.get_classification_model()
print(model.summary())
model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64)

model.save("Models/Model14.keras")


