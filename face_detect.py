import cv2
import matplotlib.pyplot as plot

# Load the cascade
"pre-trained model for frontal face detection from the OpenCV"
haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Read the input image
image = cv2.imread('images/messi.jpg')

# Convert into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
all_faces = haar_cascade_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

# print the no. of faces found
print('Faces found: ', len(all_faces))

# drawing the rectangle over the face to clearly pointing it out in the image
for (x, y, w, h) in all_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image
cv2.imshow('image', image)
plot.imshow(all_faces)

# for exiting the program
print("Press any key in the image to exit.")
cv2.waitKey()
