import tensorflow as tf

from tensorflow.keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K
import numpy as np
import csv

# read csv file
# first column is image dataset, second column is labels dataset

csv_path = "/home/tefub/Downloads/Set1_preprocessed/Set1_preprocessed/labels_dataset.csv"

images_dataset = []
labels_dataset = []

with open(csv_path, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # skips the first line of the csv, as this is the column headers

    # Populates the image_dataset and labels_dataset from the premade CSV file
    for row in csv_reader:
        images_dataset.append(row[0])
        labels_dataset.append(row[1])

def create_model():
    # Load ResNet50 as the base model, excluding top layers
    # base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
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

def generate_train_image_pairs(images_dataset, labels_dataset):
    # Group image indices by their labels
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = dict()
    for label in unique_labels:
        label_wise_indices.setdefault(label,
                                      [index for index, curr_label in enumerate(labels_dataset) if
                                       label == curr_label])
    
    pair_images = []
    pair_labels = []

    # Create positive and negative image pairs
    for index, image in enumerate(images_dataset):
        # Positive pair: same label
        pos_indices = label_wise_indices.get(labels_dataset[index])
        pos_image = images_dataset[np.random.choice(pos_indices)]
        pair_images.append((image, pos_image))
        pair_labels.append(1)

        # Negative pair: different label
        neg_indices = np.where(labels_dataset != labels_dataset[index])
        neg_image = images_dataset[np.random.choice(neg_indices[0])]
        pair_images.append((image, neg_image))
        pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)

def generate_test_image_pairs(images_dataset, labels_dataset, image):
    # Group image indices by labels
    unique_labels = np.unique(labels_dataset)
    label_wise_indices = {label: [index for index, curr_label in enumerate(labels_dataset) if label == curr_label]
                          for label in unique_labels}

    pair_images = []
    pair_labels = []

    # Create pairs with the test image and one image from each label group
    for label, indices_for_label in label_wise_indices.items():
        test_image = images_dataset[np.random.choice(indices_for_label)]
        pair_images.append((image, test_image))
        pair_labels.append(label)

    return np.array(pair_images), np.array(pair_labels)

# Create the feature extractor model and generate feature vectors for both images
feature_extractor = create_model()
imgA = Input(shape=(224, 224, 1))
imgB = Input(shape=(224, 224, 1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)

# Lambda layer to calculate distance
distance = Lambda(euclidean_distance)([featA, featB])

# Output layer for similarity score
outputs = Dense(1, activation="sigmoid")(distance)

# Compile the Siamese model using binary cross-entropy
model = Model(inputs=[imgA, imgB], outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Model training
images_pair, labels_pair = generate_train_image_pairs(images_dataset, labels_dataset)
history = model.fit([images_pair[:, 0], images_pair[:, 1]], labels_pair[:],validation_split=0.1,batch_size=64,epochs=100)

# Select random test image and generate test pairs
image = images_dataset[92] #?? 
test_image_pairs, test_label_pairs = generate_test_image_pairs(images_dataset, labels_dataset, image) 

# Predict similarity for each test image pair
threshold = 0.5
predictions = []
for index, pair in enumerate(test_image_pairs):
    pair_image1 = np.expand_dims(pair[0], axis=-1)
    pair_image1 = np.expand_dims(pair_image1, axis=0)
    pair_image2 = np.expand_dims(pair[1], axis=-1)
    pair_image2 = np.expand_dims(pair_image2, axis=0)
    
    prediction = model.predict([pair_image1, pair_image2])[0][0]  # Predict similarity score
    predictions.append(prediction)
    
    # Determine if the pair is similar or not
    if prediction > threshold:
        print(f"Pair {index + 1}: Similar (Score: {prediction:.4f})")
    else:
        print(f"Pair {index + 1}: Not Similar (Score: {prediction:.4f})")
