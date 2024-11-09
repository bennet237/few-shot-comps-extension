# Code that works on Luke's computer
# Proof of concept that shows preprocessing which identifies the face,
# crops the image, rotates the image so the face is upright, and
# resizes the image to be 224x224

# apt install cmake, need admin permissions on linux machine for this
# pip install dlib
# pip install opencv-python
# pip install matplotlib
# pip install numpy

import os
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_path = '/home/drakel2/Desktop/TuftsFaces/Set2/30//TD_RGB_E_4.jpg' # change path to access images from other places
    display_boxed_image = True # if true, displays whole image with face. if false, displays cropped and rotated image.

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # dlib only works with black and white images

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

    if display_boxed_image: # display image with a box around the image

        image_with_box = image_rgb.copy()

        # Create a semi-transparent overlay for the highlight
        overlay = image_with_box.copy()
        cv2.rectangle(overlay, 
                    (x, y),  # top-left corner
                    (x + w, y + h),  # bottom-right corner
                    (0, 255, 0),  # bright green in RGB
                    -1)  # Fill the rectangle
        
        # Apply the transparent overlay
        alpha = 0.2  # Transparency factor (0 = transparent, 1 = opaque)
        image_with_box = cv2.addWeighted(overlay, alpha, image_with_box, 1 - alpha, 0)
        
        # Draw the thick green border
        cv2.rectangle(image_with_box, 
                    (x, y),  # top-left corner
                    (x + w, y + h),  # bottom-right corner
                    (0, 255, 0),  # bright green in RGB
                    8)  # Increased thickness of the line
        
        # save as a jpg for poster
        # cv2.imwrite('face_with_box.jpg', cv2.cvtColor(image_with_box, cv2.COLOR_RGB2BGR))

        # Display the image using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_box)
        plt.axis('off')  # Hide axes
        plt.show()

    else: # display the cropped and rotated image

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
        rotated_image = cv2.warpAffine(image_rgb, M, (image_rgb.shape[1], image_rgb.shape[0])) # try to change to image_rgb

        # Reduce the image so it is only cropped
        rotated_cropped_face = rotated_image[y:y+h, x:x+w]

        # Resize the cropped face to the desired size of 224x224
        target_size = (224, 224)
        resized_face = cv2.resize(rotated_cropped_face, target_size)

        # Display cropped and rotated face
        plt.imshow(resized_face)
        plt.axis('off')
        plt.show()