import cv2
import keras.models
import numpy as np

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = img / 255
    return img

model = keras.models.load_model('trained/')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
  ret, frame = cap.read()
  frame = preprocessing(frame)
  img = frame
  frame = cv2.resize(frame, (400,400))
  cv2.imshow('frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  img = np.array(img)
  img = img.reshape(1, 100, 100, 1)
  predictions = model.predict(img)
  klasa = np.argmax(predictions, axis=1)

  print(img.shape, type(img), klasa, np.amax(predictions))


cap.release()
cv2.destroyAllWindows()