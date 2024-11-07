# luke's attempt at a siamese model
# will try implementing it off of some different source code
# also will code parts of it from scratch

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/my-repo --rm nvcr.io/nvidia/tensorflow:23.10-tf2-py3
# cd /my-repo/preprocessing
# pip install tensorflow==2.17.1

# pip install tensorflow-addons

# pip install opencv-python-headless
# python luke_siamese.py

import tensorflow as tf
import tensorflow_addons as tfa # pip install tensorflow-addons
import numpy as np
import csv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report


image_directory = "TuftsFaces/Sets1-4_preprocessed/" # update this with appropriate path if using different folder
csv_path = image_directory + "labels_dataset_no_shades.csv" # change depending on dataset you want to use.

# Data preparation
def load_and_preprocess_image(image_path):
    """Load and preprocess image for ResNet50"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads in BGR, so need to convert
    img = img.astype(np.float32) / 255.0 # normalizes image, giving proper values.
    return img

def read_csv_data(csv_path):
    """Read data from CSV file using csv package"""
    image_paths = [] # file paths
    identities = [] # corresponding "id number" saying the person corresponding to each person
    
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # skip first row of csv, as this is csv header
        for row in csv_reader:
            full_image_path = image_directory + row[0] # appends full directory file path
            image_paths.append(full_image_path)
            identities.append(row[1])
    
    return image_paths, identities

def create_pairs(image_paths, identities, positive_pairs_per_person=1):
    """Create positive and negative pairs for training
    
    Args:
        image_paths: List of image file paths
        identities: List of corresponding identity numbers
        positive_pairs_per_person: Number of positive pairs to create per person
                                if no positive pairs variable given, sets it to one
    """
    pairs = [] # positive and negative image pair file paths
    labels = [] # 0 or 1, depending on whether it is a positive (1) or negative (0) pair
    
    # Create identity to images mapping
    identity_to_images = defaultdict(list)
    for path, identity in zip(image_paths, identities):
        identity_to_images[identity].append(path)
    
    # For each identity
    for identity, paths in identity_to_images.items():
        if len(paths) < 2:
            continue  # Skip if not enough images for this person
            
        # Create all possible positive pairs for this person (shouldn't be very many)
        all_possible_pairs = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                all_possible_pairs.append([paths[i], paths[j]])
                
        # Randomly select the required number of unique positive pairs
        # Will either select all possible positive pairs, or number of pairs user enters
        num_pairs = min(positive_pairs_per_person, len(all_possible_pairs))
        selected_positive_pairs = np.random.choice(
            len(all_possible_pairs), 
            size=num_pairs, 
            replace=False  # This ensures no duplicate pairs are selected
        )
        
        # Add the selected positive pairs
        for idx in selected_positive_pairs:
            pairs.append(all_possible_pairs[idx])
            labels.append(1)
        
        # Create negative pairs (one for each positive pair to maintain balance)
        other_paths = []
        for other_identity, other_imgs in identity_to_images.items():
            if other_identity != identity:
                other_paths.extend(other_imgs)
        
        # Create same number of negative pairs as positive pairs
        neg_samples = np.random.choice(other_paths, size=num_pairs, replace=False)
        for i in range(num_pairs):
            # Randomly select an image from the current person's images
            current_img = np.random.choice(paths)
            pairs.append([current_img, neg_samples[i]])
            labels.append(0)
    
    # Shuffle the pairs and labels together
    # very important to shuffle, as otherwise pairs appear alternating positive and negative
    # model could then learn the order, instead of actual faces
    indices = np.arange(len(pairs))
    np.random.shuffle(indices) 
    pairs = np.array(pairs)[indices]
    labels = np.array(labels)[indices]
    
    return pairs, labels

def create_base_network():
    """Create the base network using ResNet50"""
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # should be able to use global pooling average, reduces spatial information to a single vector
    # lose some info, but then takes a lot fewer parameters
    
    # First, freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers for facial verification
    x = base_model.output

    # Single larger dense block (could remove this as well and only have one dense block)
    x = Dense(512)(x) # 512 neurons
    x = tf.keras.layers.BatchNormalization()(x) # normalizes activations of neurons
    x = tf.keras.layers.ReLU()(x) # activation function
    x = tf.keras.layers.Dropout(0.2)(x) # randomly drops 20% of connections to prevent overfitting, helps generalization
    
    # Final embedding
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # FROZE CODE SO IT CAN RUN, COULD LOOK INTO THIS IN THE FUTURE
    # TOO COMPLEX OF A BASE NETWORK, UNNECESSARY CONSIDERING RESNET IS PERFORMING FEATURE EXTRACTION

    # Unfreeze the last few blocks of ResNet50
    # ResNet50 has 5 blocks, let's unfreeze the last 2 blocks
    # for layer in base_model.layers:
    #     if 'conv5' in layer.name or 'bn5' in layer.name:  # Last block
    #         layer.trainable = True
    #     if 'conv4' in layer.name or 'bn4' in layer.name:  # Second-to-last block
    #         layer.trainable = True
    
    # # First dense block
    # x = Dense(1024)(x) # 1024 neurons
    # x = tf.keras.layers.BatchNormalization()(x) # normalizes activations of neurons
    # x = tf.keras.layers.ReLU()(x) # activation function
    # x = tf.keras.layers.Dropout(0.3)(x) # randomly drops 30% of connections to prevent overfitting, helps generalization
    
    # # Second dense block
    # x = Dense(512)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
    
    # # Final dense block
    # x = Dense(256)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    return Model(inputs=base_model.input, outputs=x)

def euclidean_distance(vectors):
    """Compute euclidean distance between vectors"""
    # x = first set of embeddings
    # y = second set of embeddings
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-7))

# could implement the official TF contrastive_loss function which would be as follows
    # import tensorflow_addons as tfa # pip install tensorflow-addons

    # model.compile(
    #     loss=tfa.losses.ContrastiveLoss(margin=1.0),
    #     optimizer=Adam(),
    #     metrics=['accuracy']
    # )


def contrastive_loss(margin=1.0): # Could change the val in the future, was 2.0 before
    """Contrastive loss function with margin"""
    def loss(y_true, y_pred):
        # y_true is the 0 or 1 whether they are identical
        # y_pred is the euclidian distance between face pairs
        y_true = tf.cast(y_true, tf.float32) 
        
        # Square of the distance
        square_pred = tf.square(y_pred) # makes distance positive
        
        # Loss for similar pairs - want distance to be small
        positive_loss = y_true * square_pred
        
        # Loss for dissimilar pairs - want distance to be at least margin
        negative_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))

        # Needs to take into account both positive and negative loss, as networks 
        # need to minimize distances for the same people and maximize differences between different people
        
        # Return mean loss
        return tf.reduce_mean(positive_loss + negative_loss) / 2 # may be overkill here, as it should be a double already

    return loss

def create_siamese_network():
    """Create the complete siamese network"""
    input_shape = (224, 224, 3) # 224x224, and 3 channels being RGB
    
    base_network = create_base_network()
    
    input_a = Input(shape=input_shape) # first image in pair
    input_b = Input(shape=input_shape) # second image in pair
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # L2 normalize the embeddings
    processed_a = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(processed_a)
    processed_b = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(processed_b)
    
    # Calculates distance between the embeddings
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    
    return Model(inputs=[input_a, input_b], outputs=distance)

# Created network works as follows 
# distance = model([person1_img1, person1_img2])  # Small distance (same person)
# distance = model([person1_img1, person2_img1])  # Large distance (different people)

def train_model(image_paths, identities, epochs=10, batch_size=32, learning_rate=1e-5, positive_pairs_per_person=1):
    """Train the siamese network"""
    indices = np.arange(len(image_paths))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2,
        shuffle=True,
        random_state=42 # change this val for seeds
    )
    
    train_paths = [image_paths[i] for i in train_idx]
    train_identities = [identities[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_identities = [identities[i] for i in test_idx]
    
    train_pairs, train_labels = create_pairs(train_paths, train_identities, positive_pairs_per_person)
    test_pairs, test_labels = create_pairs(test_paths, test_identities, positive_pairs_per_person)
    
    model = create_siamese_network()
    
    model.compile(
        loss=contrastive_loss(margin=1.0),
        optimizer=Adam(learning_rate=learning_rate), # will likely still have to use learning rate decay, not built in...
        metrics=["accuracy", "precision", "recall"]
    )
    # model.compile(
    #     loss=tfa.losses.ContrastiveLoss(margin=1.0),
    #     optimizer=Adam(),
    #     metrics=["accuracy", "precision", "recall"]
    # )

    # contrastive loss seems to be better than binary-cross entropy for this task of computing image distance
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"])
    
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)
    
    train_pairs_0 = np.array([load_and_preprocess_image(img) for img in train_pairs[:, 0]])
    train_pairs_1 = np.array([load_and_preprocess_image(img) for img in train_pairs[:, 1]])
    test_pairs_0 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 0]])
    test_pairs_1 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 1]])
    
    # Learning rate schedule with warmup
    initial_learning_rate = learning_rate
    warmup_epochs = 2
    total_steps = epochs * (len(train_pairs) // batch_size)
    warmup_steps = warmup_epochs * (len(train_pairs) // batch_size)
    
    def warmup_cosine_decay_schedule(step): # adam has its own function, which may be of more use... see how to use this.
        # Warmup phase
        if step < warmup_steps:
            return initial_learning_rate * (step / warmup_steps)
        # Cosine decay phase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return initial_learning_rate * 0.5 * (1 + tf.cos(tf.constant(np.pi) * progress))
    
    callbacks = [
        # Custom learning rate scheduler
        tf.keras.callbacks.LearningRateScheduler(warmup_cosine_decay_schedule),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    ]
    
    history = model.fit( # look into model.fit, model.compile, and model.predict
        [train_pairs_0, train_pairs_1],
        train_labels,
        validation_data=([test_pairs_0, test_pairs_1], test_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

# # Main execution
# if __name__ == "__main__":
#     # Load data from CSV
#     image_paths, identities = read_csv_data(csv_path)
    
#     # Train the model
#     model, history = train_model(image_paths, identities, positive_pairs_per_person=2) # selects two positive pairs per person, can change later
    
#     # Save the model
#     # model.save('siamese_face_verification.h5')

def analyze_model_predictions(model, test_pairs_0, test_pairs_1, test_labels, threshold=1.0):
    """
    Analyze model predictions and create visualizations to understand model behavior
    """
    # Get model predictions (distances)
    distances = model.predict([test_pairs_0, test_pairs_1])
    
    # Convert distances to binary predictions using threshold
    predictions = (distances < threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predictions) # not sure how this is used
    
    # Create detailed classification report
    report = classification_report(test_labels, predictions, output_dict=True)
    
    # Print statistics
    print("\nModel Statistics:")
    print(f"Average distance for same person pairs (Class 1): {distances[test_labels == 1].mean():.4f}")
    print(f"Average distance for different person pairs (Class 0): {distances[test_labels == 0].mean():.4f}")
    print(f"\nClassification Report:\n")
    for label, metrics in report.items():
        if label in ['0', '1']:
            print(f"Class {label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1-score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}") # may not work. we will see.
    
    return distances, predictions, report

def find_optimal_threshold(distances, test_labels):
    """
    Find the optimal threshold that maximizes accuracy
    """
    thresholds = np.linspace(0, 2, 200)
    accuracies = []
    
    for threshold in thresholds:
        predictions = (distances < threshold).astype(int)
        accuracy = (predictions == test_labels).mean()
        accuracies.append(accuracy)
    
    optimal_threshold = thresholds[np.argmax(accuracies)]
    
    
    return optimal_threshold

def test_specific_pairs(model, image_paths, identities, num_pairs=5):
    """
    Test model on specific pairs to visualize its behavior
    """
    # Create some same-person pairs
    same_pairs = []
    diff_pairs = []
    
    # Group images by identity
    id_to_imgs = defaultdict(list)
    for path, id_ in zip(image_paths, identities):
        id_to_imgs[id_].append(path)
    
    # Get same-person pairs
    for id_, paths in id_to_imgs.items():
        if len(paths) >= 2:
            same_pairs.append((paths[0], paths[1], 1))
            if len(same_pairs) >= num_pairs:
                break
    
    # Get different-person pairs
    ids = list(id_to_imgs.keys())
    for i in range(num_pairs):
        id1, id2 = np.random.choice(ids, 2, replace=False)
        diff_pairs.append((id_to_imgs[id1][0], id_to_imgs[id2][0], 0))
    
    # Combine and process pairs
    test_pairs = same_pairs + diff_pairs
    paths_1 = [p[0] for p in test_pairs]
    paths_2 = [p[1] for p in test_pairs]
    labels = [p[2] for p in test_pairs]
    
    # Load and preprocess images
    imgs_1 = np.array([load_and_preprocess_image(p) for p in paths_1])
    imgs_2 = np.array([load_and_preprocess_image(p) for p in paths_2])
    
    # Get predictions
    distances = model.predict([imgs_1, imgs_2])
    

if __name__ == "__main__":
    # Load data from CSV
    image_paths, identities = read_csv_data(csv_path)

    # Maybe put the assortment part in the main function...
    
    # Train the model
    model, history = train_model(image_paths, identities, positive_pairs_per_person=2)
    
    # Get test data
    _, test_idx = train_test_split(np.arange(len(image_paths)), test_size=0.2, random_state=5)
    test_paths = [image_paths[i] for i in test_idx]
    test_identities = [identities[i] for i in test_idx]
    test_pairs, test_labels = create_pairs(test_paths, test_identities, positive_pairs_per_person=2)
    
    # Prepare test data
    test_pairs_0 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 0]])
    test_pairs_1 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 1]])
    
    # Analyze model predictions
    distances, predictions, report = analyze_model_predictions(model, test_pairs_0, test_pairs_1, test_labels)
    distances, predictions, report = analyze_model_predictions(model, test_pairs_0, test_pairs_1, test_labels) # change this for the training set stff
    distances, predictions, report = analyze_model_predictions(model, test_pairs_0, test_pairs_1, test_labels) # change this as well.
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(distances, test_labels)
    
    # Test on specific pairs
    # test_specific_pairs(model, image_paths, identities)

    # get function to return training and validation pairs
    # insert optimal threshold into the training function
    # figure out what the model even returns for the training, as it seems that the threshold may just be at 1.
