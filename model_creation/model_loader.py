import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.optimizers import Adam

# Backbone archictecture used
ARCHITECTURE = "VGG19"  # or "ResNet50"

# what you use to load the weights. make sure you're using the same arhictecture that the model was trained on
MODEL_PATH = "saved_models/generated_model.keras"

# Define the required custom functions so that the model can be loaded properly
# these are the same functions for when the model was initialzied in luke_siamese.py
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-7))

def contrastive_loss(margin=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        positive_loss = y_true * square_pred
        negative_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(positive_loss + negative_loss) / 2
    return loss

def create_base_network(architecture="VGG19"):
    """Create base network with specified architecture"""
    if architecture == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    elif architecture == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError("Architecture must be either 'VGG19' or 'ResNet50'")
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return Model(inputs=base_model.input, outputs=x)

def create_siamese_network(architecture="VGG19"):
    input_shape = (224, 224, 3)
    base_network = create_base_network(architecture)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    processed_a = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(processed_a)
    processed_b = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(processed_b)
    
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    
    return Model(inputs=[input_a, input_b], outputs=distance)

if __name__ == "__main__":
    # Create the model with specified architecture
    print(f"Creating model with {ARCHITECTURE} architecture...")
    model = create_siamese_network(ARCHITECTURE)
    
    # Compile the model
    model.compile(
        loss=contrastive_loss(margin=0.5),
        optimizer=Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )
    
    # Load the pretrained weights
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_weights(MODEL_PATH)
    
    print("Successfully loaded model!")

    # now from here, you're able to test the model however you would like!
    # you're able to feed the model any images and compute the distance, treating the model similarily as to how one would in the other files