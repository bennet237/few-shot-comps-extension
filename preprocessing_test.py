# works in google collab, will need to change file names

# Code that works on Luke's computer, paths have been changed such that now it hopefully works

import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image_path = '/Users/luke/Downloads/IMG_0977.jpg'
image_path = '/content/sample_data/Face_test.jpg'

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ML model to detect faces
detector = dlib.get_frontal_face_detector()
# ML model to predict where facial features are
# landmark_model_file_path = '/Users/luke/Downloads/shape_predictor_68_face_landmarks.dat'
landmark_model_file_path = '/content/sample_data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmark_model_file_path)

# detects faces in the image (could be multiple but in our case one)
faces = detector(gray)
face = faces[0]


x, y, w, h = (face.left(), face.top(), face.width(), face.height())

# Gets facial landmarks (eyes, nose, mouth etc.)
landmarks = predictor(gray, face)

# Get coordinates of the eyes (landmarks: 36 for left eye, 45 for right eye)
left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

# Calculate the angle between the eyes
dy = right_eye[1] - left_eye[1]
dx = right_eye[0] - left_eye[0]
angle = np.degrees(np.arctan2(dy, dx))

# Get the center of the face for rotation
center = (x + w // 2, y + h // 2)

# Rotate the image around the center of the face
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Reduce the image so it is only cropped
rotated_cropped_face = rotated_image[y:y+h, x:x+w]

# Display cropped and rotated face
plt.imshow(cv2.cvtColor(rotated_cropped_face, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
