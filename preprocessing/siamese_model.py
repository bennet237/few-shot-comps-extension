import tensorflow as tf # to load tf, use "pip install tensorflow==2.17.1"
from keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.applications import ResNet50
from sklearn.model_selection import train_test_split
import tensorflow.python.keras.backend as K
import numpy as np
import csv
import os
import cv2 # for image processing "pip install opencv-python-headless"


# read csv file
# first column is image dataset, second column is labels dataset
image_directory = "TuftsFaces/Sets1-4_preprocessed/" # update this with appropriate path

csv_path = image_directory + "labels_dataset.csv" # should not have to change this

images_dataset = []
labels_dataset = []

with open(csv_path, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # skips the first line of the csv, as this is the column headers

    # Populates the image_dataset and labels_dataset from the premade CSV file
    for row in csv_reader:
        correct_image_path = image_directory + row[0] # update the appropriate image path
        images_dataset.append(correct_image_path)
        labels_dataset.append(row[1])

# print(images_dataset) # an array with all of the file paths to the JPG images
# print(labels_dataset) # corresponding array which says which person (label) each image path belongs to

# Loads the preprocesed images from the given filepath
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read it in as grayscale, as that is what ML models work on
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    return image

def create_model():
    # Load ResNet50 as the base model, excluding top layers
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Freeze base model layers to retain pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    # Input layer for grayscale images
    inputs = Input(shape=(224, 224, 1))
    x = tf.keras.layers.Conv2D(3, (1, 1))(inputs)  # Convert grayscale to 3-channel input
    x = base_model(x)

    # Feature extraction layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)  # Regularization to prevent overfitting

    return Model(inputs, x)

# Compute the Euclidean distance between the two feature vectors
def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

# Split dataset into 70% training and 30% testing
train_images, test_images, train_labels, test_labels = train_test_split(
    images_dataset,
    labels_dataset, 
    test_size=0.3,
    random_state=42
)

def generate_train_image_pairs(train_images, train_labels):
    # Group image indices by their labels
    unique_labels = np.unique(train_labels)
    label_wise_indices = {label: [i for i, lbl in enumerate(train_labels) if lbl == label] for label in unique_labels}

    pair_images = []
    pair_labels = []

    for i, image_path in enumerate(train_images):
        image = load_image(image_path)

        # Positive pair, same person
        pos_indices = label_wise_indices[train_labels[i]]
        pos_image_path = train_images[np.random.choice(pos_indices)]
        pos_image = load_image(pos_image_path)  # Load positive image
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        # Negative pair, different people
        neg_indices = np.where(train_labels != train_labels[i])[0]
        neg_image_path = train_images[np.random.choice(neg_indices)]
        neg_image = load_image(neg_image_path)  # Load negative image
        pair_images.append((image, neg_image))
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)


def generate_test_image_pairs(images_dataset, labels_dataset, test_images, test_labels):
    # Group image indices by labels
    unique_labels = np.unique(test_labels)
    label_wise_indices = {label: [index for index, curr_label in enumerate(labels_dataset) if label == curr_label]
                          for label in unique_labels}

    pair_images = []
    pair_labels = []

    # Loop over each test image and label
    for i, test_image_path in enumerate(test_images):
        test_image = load_image(test_image_path)
        test_label = test_labels[i]

        # Positive pair, same label in test set
        pos_indices = label_wise_indices[test_label]
        pos_image_path = images_dataset[np.random.choice(pos_indices)]
        pos_image = load_image(pos_image_path)
        pair_images.append((test_image, pos_image))
        pair_labels.append(1)  # Similarity label for positive pair

        # Negative pair, different label
        neg_label = np.random.choice([label for label in unique_labels if label != test_label])
        neg_indices = label_wise_indices[neg_label]
        neg_image_path = images_dataset[np.random.choice(neg_indices)]
        neg_image = load_image(neg_image_path)
        pair_images.append((test_image, neg_image))
        pair_labels.append(0)  # Dissimilarity label for negative pair

    return np.array(pair_images), np.array(pair_labels)

# Create the feature extractor model and generate feature vectors for both images
feature_extractor = create_model()
imgA = Input(shape=(224, 224, 1))
imgB = Input(shape=(224, 224, 1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)

# Lambda layer to calculate distance
distance = Lambda(euclidean_distance, output_shape=(1,))([featA, featB])

# Output layer for similarity score
outputs = Dense(1, activation="sigmoid")(distance)

# Compile the Siamese model using binary cross-entropy
model = Model(inputs=[imgA, imgB], outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"])

# Model training
# Fails here consistently, likely due to the cause that the images are not loaded.
images_pair, labels_pair = generate_train_image_pairs(images_dataset, labels_dataset)
# Can change epochs later with appropriate amounts
history = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair, validation_split=0.1, batch_size=64, epochs=10)

# Example: Test image pairs
test_image_pairs, test_label_pairs = generate_test_image_pairs(images_dataset, labels_dataset, test_images, test_labels)

# Model Evaluation: Evaluate the model's performance
test_loss, test_accuracy, test_precision, test_recall = model.evaluate([test_image_pairs[:, 0], test_image_pairs[:, 1]], test_label_pairs)

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")


# CURRENTLY NOT WORKING, CAN FIX AND UNCOMMENT IN THE FUTURE
# PRINT STATEMENTS DO NOTHING AT THE MOMENT...
# Predict similarity for each test image pair
# threshold = 0.5
# predictions = []
# for index, pair in enumerate(test_image_pairs):
#     pair_image1 = np.expand_dims(pair[0], axis=-1)
#     pair_image1 = np.expand_dims(pair_image1, axis=0)
#     pair_image2 = np.expand_dims(pair[1], axis=-1)
#     pair_image2 = np.expand_dims(pair_image2, axis=0)
    
#     prediction = model.predict([pair_image1, pair_image2])[0][0]  # Predict similarity score
#     predictions.append(prediction)
    
#     # Determine if the pair is similar or not
#     if prediction > threshold:
#         print(f"Pair {index + 1}: Similar (Score: {prediction:.4f})")
#     else:
#         print(f"Pair {index + 1}: Not Similar (Score: {prediction:.4f})")
