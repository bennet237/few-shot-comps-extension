# Creates a CSV file with all of the labels of images
# so that Siamese model can generate positive and negative pairs of images

# Goes off of the assumption that the image dataset is laid out
# as described in the preprocessing. Structure pictured below.
# - Input folder
#    - Person 1
#       - Image 1.jpg
#       - Image 2.jpg
#    - Person 2
#       - Image 3.jpg
#       - Image 4.jpg

import os
import csv

input_folder_path = '/home/drakel2/Desktop/Tufts Faces/Set1_preprocessed' # modify for approriate path name

def create_labels_dataset(input_folder_path, output_csv_path):
    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Header row of CSV
        writer.writerow(['Image Path', 'Person Label'])

        # Iterate over each person's folder in the input folder
        for person_folder in os.listdir(input_folder_path):
            person_folder_path = os.path.join(input_folder_path, person_folder)

            # Ensure it's a directory (person's folder)
            if os.path.isdir(person_folder_path):
                # Iterate over each image in the person's folder
                for image_name in os.listdir(person_folder_path):
                    image_path = os.path.join(person_folder_path, image_name)

                    # Check if it's a .jpg file
                    if os.path.isfile(image_path) and image_name.lower().endswith('.jpg'):
                        # Write image path and associated person label to CSV
                        writer.writerow([image_path, person_folder])

    print(f"Labels dataset saved to {output_csv_path}")


if __name__ == "__main__":
    # Output CSV path, saved within input folder
    output_csv_path = os.path.join(input_folder_path, 'labels_dataset.csv')

    # Call the function to create the labels dataset
    create_labels_dataset(input_folder_path, output_csv_path)
