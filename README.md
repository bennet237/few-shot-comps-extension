# Siamese Neural Networks for Few-Shot Facial Image Recognition

## Overview
This project implements a few-shot learning approach using Siamese Neural Networks for facial verification tasks. Unlike traditional deep learning models that require massive datasets, this implementation can effectively learn from minimal examples, making it particularly valuable for applications where data collection is constrained by privacy concerns or practical limitations.

## Key Features
- Few-shot learning capability requiring minimal training examples per person
- Siamese Neural Network architecture with shared weights
- Support for various pre-trained models (VGG19, ResNet50, VGG16)
- Robust preprocessing pipeline for facial image standardization
- Handling of different facial orientations and occlusions
- Customizable positive/negative pair generation

## Dataset
The model was trained and tested on the Tufts-Face-Database, which includes:
- 112 individuals (74 females + 38 males)
- Age range: 4-70 years
- 13 images per person:
  - 5 facial expression images (neutral, smile, eyes closed, shocked, wearing shades)
  - 8 orientation images (different angles from left to right)

## Technical Implementation

### Preprocessing Pipeline
1. Face detection using Dlib
2. Facial landmark identification (68 points)
3. Image rotation for upright face alignment
4. Cropping to square aspect ratio
5. Resizing to 224x224 pixels

### Model Architecture
- Base: Pre-trained VGG19 (performed best among tested models)
- Custom layers:
  - Two dense blocks (512 units each)
  - Batch normalization
  - ReLU activation
- L2 normalization for feature vectors
- Euclidean distance computation
- Contrastive loss function

## Key Findings
1. Model Performance:
   - Best accuracy: 96% (without occlusions/orientations)
   - AUC: 0.99

2. Pre-trained Model Comparison:
   - VGG19: 96% accuracy
   - VGG16: 92% accuracy
   - ResNet50: 79% accuracy

3. Impact of Image Variations:
   - Best performance achieved with frontal-facing images without occlusions
   - Decreased performance with shaded images and varying orientations
   - False negatives primarily occur with occluded faces
   - False positives occur with similar facial features across different individuals

## Dependencies
- Python
- OpenCV
- Dlib
- TensorFlow/Keras
- NumPy
- Pandas

## Usage
It is suggested that after you complete the preprocessing step, it is best to move the image folder with all of the preprocessed images into the experiments tab.

## Future Improvements
1. Increase occlusion training data
2. Experiment with alternative loss functions (Triplet Loss, Binary Cross-Entropy)
3. Optimize false positive/negative reduction
4. Enhance robustness to facial occlusions

## Citation
If you use this work in your research, please cite:
```
Drake, L., & Tefu, B. (2024). Siamese Neural Networks for Few-Shot Facial Image Recognition.
```

## Contact
If you have any further inquiries, please reach out to either drakel2@carleton.edu or tefub@carleton.edu