from luke_siamese import euclidean_distance
from tensorflow.keras.models import load_model

model, history, train_idx, test_idx, val_idx = load_model('siamese_face_verification_shades2.keras', safe_mode=False)