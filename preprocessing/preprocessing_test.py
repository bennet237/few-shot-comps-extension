# Code that works on Luke's computer
# Proof of concept that shows preprocessing which identifies the face,
# crops the image, rotates the image so the face is upright, and
# resizes the image to be 224x224

import os
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_path = '/Users/luke/Downloads/TD_RGB_E_Set1/23/TD_RGB_E_5.jpg' # change path to access images from other places

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ML model to detect faces
    detector = dlib.get_frontal_face_detector()

    # ML model to predict where facial features are
    current_file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    predictor = dlib.shape_predictor(os.path.join(current_file_path, 'shape_predictor_68_face_landmarks.dat'))

    # detects faces in the image
    list_of_faces = detector(gray)

    # since our images are only ever of one person, only need first item of list
    face = list_of_faces[0]

    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    # Gets facial landmarks (eyes, nose, mouth, etc.)
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

    # Resize the cropped face to the desired size of 224x224
    target_size = (224, 224)
    resized_face = cv2.resize(rotated_cropped_face, target_size)

    # Display cropped and rotated face
    plt.imshow(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()