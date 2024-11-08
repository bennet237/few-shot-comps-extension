# luke's attempt at a siamese model
# will try implementing it off of some different source code
# also will code parts of it from scratch

# dataset info: 112 people, 4 images per person
# (Could get more images, removed sunglasses)

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/my-repo --rm nvcr.io/nvidia/tensorflow:23.10-tf2-py3
# cd /my-repo/preprocessing
# pip install tensorflow==2.17.1
# pip install opencv-python-headless
# python luke_siamese.py

import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc


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

    # may need to add the extra layers back or something... 

    # Single larger dense block (could remove this as well and only have one dense block)
    x = Dense(512)(x) # 512 neurons
    x = tf.keras.layers.BatchNormalization()(x) # normalizes activations of neurons
    x = tf.keras.layers.ReLU()(x) # activation function
    x = tf.keras.layers.Dropout(0.2)(x) # randomly drops 20% of connections to prevent overfitting, helps generalization
    
    # Final embedding
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return Model(inputs=base_model.input, outputs=x)

def euclidean_distance(vectors):
    """Compute euclidean distance between vectors"""
    # x = first set of embeddings
    # y = second set of embeddings
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-7))


# need a custom contrastive loss function, as old TF contrastive_loss builtin function was discontinued and doesn't
# work on the version of TF that we are using.
def contrastive_loss(margin=0.5): # Can play around with this and change it
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
        return tf.reduce_mean(positive_loss + negative_loss) / 2 # reduce_mean be overkill here, as it should be a double already

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
    # Not sure what this means, but models I've looked at include this
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
    )   # even though a random seed is here, generate pairs still operates randomly
        # may consider in the future implementing a seed into that function if necessary... (probably have to for comparative accuracy)
    
    train_paths = [image_paths[i] for i in train_idx]
    train_identities = [identities[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_identities = [identities[i] for i in test_idx]
    
    train_pairs, train_labels = create_pairs(train_paths, train_identities, positive_pairs_per_person)
    test_pairs, test_labels = create_pairs(test_paths, test_identities, positive_pairs_per_person)
    
    model = create_siamese_network()

    # Could find out somehow to tweak the accuracy... Pretty sure it falls under here...
    # In your model compilation:
    model.compile(
        loss=contrastive_loss(margin=1.0),
        optimizer=Adam(learning_rate=learning_rate), # will likely still have to use learning rate decay, not built in...
        metrics=["accuracy", "precision", "recall"]
    )

    # contrastive loss seems to be better than binary-cross entropy for this task of computing image distance
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"])
    
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)
    
    # separates first and second images from each pair, then loads and preprocesses them
    train_pairs_0 = np.array([load_and_preprocess_image(img) for img in train_pairs[:, 0]])
    train_pairs_1 = np.array([load_and_preprocess_image(img) for img in train_pairs[:, 1]])
    test_pairs_0 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 0]])
    test_pairs_1 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 1]])
    
    history = model.fit( # look into model.fit, model.compile, and model.predict
        [train_pairs_0, train_pairs_1],
        train_labels,
        validation_data=([test_pairs_0, test_pairs_1], test_labels),
        batch_size=batch_size,
        epochs=epochs
    )
    
    return model, history, test_idx

# calculates optimal threshold to maximize accuracy
def find_optimal_threshold(model, image_paths, identities, positive_pairs_per_person=1, num_thresholds=100):
    """
    Find the optimal threshold value that maximizes classification accuracy.
    
    Args:
        model: Trained Siamese network
        image_paths: List of image paths
        identities: List of corresponding identities
        positive_pairs_per_person: Number of positive pairs per person for evaluation
        num_thresholds: Number of threshold values to test
        
    Returns:
        optimal_threshold: The threshold that maximizes accuracy
        best_accuracy: The accuracy achieved at the optimal threshold
        threshold_metrics: Dictionary containing detailed metrics at the optimal threshold
    """
    # Create evaluation pairs
    pairs, true_labels = create_pairs(image_paths, identities, positive_pairs_per_person)
    
    # Load and preprocess images
    pairs_0 = np.array([load_and_preprocess_image(img) for img in pairs[:, 0]])
    pairs_1 = np.array([load_and_preprocess_image(img) for img in pairs[:, 1]])
    
    # Get model predictions (distances)
    distances = model.predict([pairs_0, pairs_1])
    
    # Initialize variables to store best results
    best_accuracy = 0
    optimal_threshold = 0
    best_metrics = None
    
    # Calculate min and max distances to set threshold range
    min_dist = np.min(distances) # get minimum distance between pairs
    max_dist = np.max(distances) # get maximum distance between pairs
    thresholds = np.linspace(min_dist, max_dist, num_thresholds) # linearly space 100 vals between min and max
    
    print("\nFinding optimal threshold...")
    print("-" * 50)
    
    # Test each threshold
    for threshold in thresholds:
        pred_labels = (distances <= threshold).astype(int)
        accuracy = np.mean(pred_labels.flatten() == true_labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_threshold = threshold
            
            # Calculate additional metrics at this threshold
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels.flatten())
            conf_matrix = confusion_matrix(true_labels, pred_labels.flatten())
            
            best_metrics = {
                'accuracy': best_accuracy,
                'threshold': optimal_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }
    
    print(f"\nOptimal threshold found: {optimal_threshold:.4f}")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    print("\nMetrics at optimal threshold:")
    print("Same Person (Class 1):")
    print(f"Precision: {best_metrics['precision'][1]:.4f}")
    print(f"Recall: {best_metrics['recall'][1]:.4f}")
    print(f"F1-Score: {best_metrics['f1'][1]:.4f}")
    print("\nDifferent Person (Class 0):")
    print(f"Precision: {best_metrics['precision'][0]:.4f}")
    print(f"Recall: {best_metrics['recall'][0]:.4f}")
    print(f"F1-Score: {best_metrics['f1'][0]:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix at optimal threshold:")
    print("True\Pred  Different  Same")
    print(f"Different  {best_metrics['confusion_matrix'][0][0]:9d} {best_metrics['confusion_matrix'][0][1]:6d}")
    print(f"Same       {best_metrics['confusion_matrix'][1][0]:9d} {best_metrics['confusion_matrix'][1][1]:6d}")
    
    return optimal_threshold, best_accuracy, best_metrics

# new method, evaluates the entire thing with all of the images
def evaluate_siamese_network(model, image_paths, identities, threshold=0.5, positive_pairs_per_person=1):
    """
    Comprehensive evaluation of Siamese network performance.
    
    Args:
        model: Trained Siamese network
        image_paths: List of image paths
        identities: List of corresponding identities
        threshold: Distance threshold for classification (default 0.5, but will have optimal_threshold inserted)
        positive_pairs_per_person: Number of positive pairs per person for evaluation
    """
    
    # Create evaluation pairs
    pairs, true_labels = create_pairs(image_paths, identities, positive_pairs_per_person)
    
    # Load and preprocess images
    pairs_0 = np.array([load_and_preprocess_image(img) for img in pairs[:, 0]])
    pairs_1 = np.array([load_and_preprocess_image(img) for img in pairs[:, 1]])
    
    # Get model predictions (distances)
    distances = model.predict([pairs_0, pairs_1])
    
    # Convert distances to binary predictions using threshold
    pred_labels = (distances <= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels.flatten())
    conf_matrix = confusion_matrix(true_labels, pred_labels.flatten())
    
    # Separate distances for same and different pairs
    same_person_distances = distances[true_labels == 1].flatten()
    diff_person_distances = distances[true_labels == 0].flatten()
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(true_labels, -distances.flatten())  # Negative distances because smaller distance = more similar
    roc_auc = auc(fpr, tpr)
    
    # Print detailed metrics
    print("\nDetailed Evaluation Metrics: (Trained on most of these)")
    print("-" * 50)
    print(f"Number of pairs evaluated: {len(true_labels)}")
    print(f"Number of same-person pairs: {len(same_person_distances)}")
    print(f"Number of different-person pairs: {len(diff_person_distances)}")
    print("\nDistance Statistics:")
    print(f"Same Person - Mean Distance: {np.mean(same_person_distances):.4f} ± {np.std(same_person_distances):.4f}")
    print(f"Different Person - Mean Distance: {np.mean(diff_person_distances):.4f} ± {np.std(diff_person_distances):.4f}")
    print(f"Same Person - Min Distance: {np.min(same_person_distances):.4f}")
    print(f"Same Person - Max Distance: {np.max(same_person_distances):.4f}")
    print(f"Different Person - Min Distance: {np.min(diff_person_distances):.4f}")
    print(f"Different Person - Max Distance: {np.max(diff_person_distances):.4f}")
    print("\nClassification Metrics (threshold = {:.2f}):".format(threshold))
    print("Same Person (Class 1):")
    print(f"Precision: {precision[1]:.4f}")
    print(f"Recall: {recall[1]:.4f}")
    print(f"F1-Score: {f1[1]:.4f}")
    print("\nDifferent Person (Class 0):")
    print(f"Precision: {precision[0]:.4f}")
    print(f"Recall: {recall[0]:.4f}")
    print(f"F1-Score: {f1[0]:.4f}")
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("True\Pred  Different  Same")
    print(f"Different  {conf_matrix[0][0]:9d} {conf_matrix[0][1]:6d}")
    print(f"Same       {conf_matrix[1][0]:9d} {conf_matrix[1][1]:6d}")
    
    return {
        'same_person_distances': same_person_distances,
        'diff_person_distances': diff_person_distances,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'auc': roc_auc
    }

# evaluates just the test set to see how it performs
def evaluate_test_set(model, test_paths, test_identities, threshold=0.5, positive_pairs_per_person=1):
    """
    Evaluate model performance specifically on the test set.
    
    Args:
        model: Trained Siamese network
        test_paths: List of image paths in the test set
        test_identities: List of corresponding identities in the test set
        threshold: Distance threshold for classification
        positive_pairs_per_person: Number of positive pairs per person for evaluation
    """
    
    print("\nTest Set Evaluation: (Not trained on these, but used in validation)")
    print("-" * 50)
    print(f"Number of test images: {len(test_paths)}")
    print(f"Number of unique identities in test set: {len(set(test_identities))}")
    
    # Create pairs only from test set
    test_pairs, test_labels = create_pairs(test_paths, test_identities, positive_pairs_per_person)
    
    # Load and preprocess test images
    test_pairs_0 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 0]])
    test_pairs_1 = np.array([load_and_preprocess_image(img) for img in test_pairs[:, 1]])
    
    # Get model predictions
    distances = model.predict([test_pairs_0, test_pairs_1])
    
    # Convert distances to binary predictions using threshold
    pred_labels = (distances <= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels.flatten())
    conf_matrix = confusion_matrix(test_labels, pred_labels.flatten())
    
    # Separate distances for same and different pairs
    same_person_distances = distances[test_labels == 1].flatten()
    diff_person_distances = distances[test_labels == 0].flatten()
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(test_labels, -distances.flatten())
    roc_auc = auc(fpr, tpr)
    
    # Print detailed metrics
    print(f"\nNumber of test pairs evaluated: {len(test_labels)}")
    print(f"Number of same-person pairs: {len(same_person_distances)}")
    print(f"Number of different-person pairs: {len(diff_person_distances)}")
    
    print("\nDistance Statistics (Test Set):")
    print(f"Same Person - Mean Distance: {np.mean(same_person_distances):.4f} ± {np.std(same_person_distances):.4f}")
    print(f"Different Person - Mean Distance: {np.mean(diff_person_distances):.4f} ± {np.std(diff_person_distances):.4f}")
    print(f"Same Person - Min Distance: {np.min(same_person_distances):.4f}")
    print(f"Same Person - Max Distance: {np.max(same_person_distances):.4f}")
    print(f"Different Person - Min Distance: {np.min(diff_person_distances):.4f}")
    print(f"Different Person - Max Distance: {np.max(diff_person_distances):.4f}")
    
    print(f"\nClassification Metrics (threshold = {threshold:.2f}):")
    print("Same Person (Class 1):")
    print(f"Precision: {precision[1]:.4f}")
    print(f"Recall: {recall[1]:.4f}")
    print(f"F1-Score: {f1[1]:.4f}")
    
    print("\nDifferent Person (Class 0):")
    print(f"Precision: {precision[0]:.4f}")
    print(f"Recall: {recall[0]:.4f}")
    print(f"F1-Score: {f1[0]:.4f}")
    
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print("True\Pred  Different  Same")
    print(f"Different  {conf_matrix[0][0]:9d} {conf_matrix[0][1]:6d}")
    print(f"Same       {conf_matrix[1][0]:9d} {conf_matrix[1][1]:6d}")
    
    accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1])
    print(f"\nAccuracy: {accuracy:.4f}")


    return {
        'same_person_distances': same_person_distances,
        'diff_person_distances': diff_person_distances,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'auc': roc_auc,
        'accuracy': accuracy
    }

# Main execution
if __name__ == "__main__":
    # Load data from CSV
    image_paths, identities = read_csv_data(csv_path)
    
    # Train the model
    desired_positive_pairs = 3 # selects three positive (and consequentially three negative) pairs per person, can change later
    # model, history = train_model(image_paths, identities, positive_pairs_per_person=desired_positive_pairs) 
    model, history, test_idx = train_model(image_paths, identities, positive_pairs_per_person=desired_positive_pairs)

    # After training
    optimal_threshold, best_accuracy, metrics = find_optimal_threshold(model, image_paths, identities, positive_pairs_per_person=desired_positive_pairs) 

    # Then use this threshold in your evaluation function
    evaluation_metrics = evaluate_siamese_network(model, image_paths, identities, threshold=optimal_threshold, positive_pairs_per_person=desired_positive_pairs)

    # Evaluate metrics on a test set
    test_paths = [image_paths[i] for i in test_idx]
    test_identities = [identities[i] for i in test_idx]
    test_metrics = evaluate_test_set(model, test_paths, test_identities, threshold=optimal_threshold, positive_pairs_per_person=desired_positive_pairs)

    # Save the model
    # model.save('siamese_face_verification.h5')

# Things to do for the future...

# try to save model, so evaluataion can save time. will also need this later.

# create separate validation from test set. Should all be unique. 70/15/15 training/validation/testing split.

# create seed function for the create_pairs method, so we are able to 

# in test, look at what it is getting right, and what it is getting wrong (specific individuals)
# could create histogram images
# look at the before training and after training statistics (get values before and after)
# make sure that training is acutally doing something!!!

# do not display GUI, save as png/jpg in separate folder. Start saving model as well.

# should set the optimal threshold just on the validation set probably and have it change as well.

# could modify the dropout and remove this; this prevents overfitting, but if our model isn't learning anything
# in the first place then this is obsolete.