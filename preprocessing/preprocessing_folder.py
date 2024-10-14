# Reads in all of the images from a folder and then stores them in a new folder in Downloads
# Performs preprocessing of cropping image, rotating, and resizing the images to be
# appropriate dimensions so that a siamese model is able to be ran upon them

import os
import dlib
import cv2
import numpy as np

# Input folder has to follow specific structure
# - Within the input folder, there will be folders for each individual person
# - Within each of the individual folders, there will be the jpg images
# Tufts dataset is currently structured this way, so if using the same dataset
# no need to modify

# Structure goes as follows...
# - Input folder
#    - Person 1
#       - Image 1.jpg
#       - Image 2.jpg
#    - Person 2
#       - Image 3.jpg
#       - Image 4.jpg

input_folder_path = '/Users/luke/Downloads/TD_RGB_E_Set1' # modify for approriate path name


# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    list_of_faces = detector(gray)
    if len(list_of_faces) == 0:
        return None  # Return None if no face is detected

    face = list_of_faces[0]  # Assuming only one face per image
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    # Get facial landmarks
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

    # Crop the rotated image to just the face
    rotated_cropped_face = rotated_image[y:y+h, x:x+w]

    # Resize the cropped face
    target_size = (224, 224)  # You can modify the target size as needed
    resized_face = cv2.resize(rotated_cropped_face, target_size)

    return resized_face

# Main function
def main():
    # Define the input folder path and create the preprocessed folder
    input_folder_name = os.path.basename(input_folder_path)
    preprocessed_folder_name = input_folder_name + '_preprocessed'
    downloads_path = os.path.expanduser("~/Downloads")
    preprocessed_folder_path = os.path.join(downloads_path, preprocessed_folder_name)

    # Create the preprocessed folder if it doesn't exist
    if os.path.exists(preprocessed_folder_path):
        raise Exception(f"Folder '{preprocessed_folder_path}' already exists.")
    else:
        os.makedirs(preprocessed_folder_path)
        print(f"Folder '{preprocessed_folder_path}' created successfully.")

    # Initialize the face detector and landmark predictor
    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    current_file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    predictor = dlib.shape_predictor(os.path.join(current_file_path, 'shape_predictor_68_face_landmarks.dat'))

    # Iterate over each person's folder in the input folder
    for person_folder in os.listdir(input_folder_path):
        person_folder_path = os.path.join(input_folder_path, person_folder)

        # Ensure it's a directory (person's folder)
        if os.path.isdir(person_folder_path):
            # Create a corresponding folder in the preprocessed folder
            preprocessed_person_folder_path = os.path.join(preprocessed_folder_path, person_folder)
            os.makedirs(preprocessed_person_folder_path, exist_ok=True)

            # Iterate over each image in the person's folder
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)

                # Check if it's a file and an image (all images in Tufts are .jpg)
                if os.path.isfile(image_path) and image_name.lower().endswith('.jpg'):
                    # Preprocess the image, using the above method
                    preprocessed_image = preprocess_image(image_path)

                    if preprocessed_image is not None:
                        # Save the preprocessed image to the corresponding folder
                        save_path = os.path.join(preprocessed_person_folder_path, image_name)
                        cv2.imwrite(save_path, preprocessed_image)
                        print(f"Preprocessed image saved: {save_path}")
                    else:
                        print(f"No face detected in {image_name}, skipping...")

# Makes sure that script only runs when directly called
if __name__ == "__main__":
    main()
