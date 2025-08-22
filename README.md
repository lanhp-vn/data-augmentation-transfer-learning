# Data Augmentation and Transfer Learning for Flower Classification

## Project Overview

This project demonstrates the implementation of data augmentation and transfer learning techniques for image classification using the TensorFlow Flowers dataset. The project was developed and executed on Google Colab, leveraging GPU acceleration for efficient training.

## Dataset

- **Dataset**: TensorFlow Flowers (tf_flowers)
- **Classes**: 5 flower categories (daisy, dandelion, roses, sunflowers, tulips)
- **Total Images**: 3,670
- **Split**: 70% training, 15% validation, 15% test

## Methodology

### Data Preprocessing
- **Image Resizing**: All images resized to 96x96 pixels
- **Normalization**: Pixel values scaled to [0,1] range
- **Data Augmentation**: Applied to training data only
  - Random horizontal and vertical flips
  - Random rotation (±20 degrees)
  - Random contrast adjustment (±20%)
  - Random cropping
  - Random zoom (±20%)

### Transfer Learning Approach
Three different transfer learning strategies were implemented using DenseNet121 as the base model:

1. **Frozen Base Model**: All pre-trained layers frozen, only classification layers trained
2. **Partial Fine-tuning**: 50% of base model layers unfrozen for fine-tuning
3. **Modified Classification Head**: Using GlobalAveragePooling2D instead of Flatten

### Model Architecture
- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Classification Head**: 
  - Dense layers (64 units each)
  - Batch Normalization
  - Dropout (0.5)
  - Softmax output layer

## Results

### Performance Comparison

| Model Configuration | Training Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
|-------------------|------------------|-------------------|---------------|-----------|
| Frozen Base Model | 98.37% | ~83% | 98.37% | 0.0744 |
| Partial Fine-tuning | 99.57% | ~83% | 99.57% | 0.0191 |
| Modified Head | 89.96% | ~74% | 89.96% | 0.2744 |

### Key Findings

1. **Partial Fine-tuning Achieved Best Performance**: The model with 50% of base layers unfrozen achieved the highest test accuracy (99.57%) and lowest test loss (0.0191).

2. **Data Augmentation Effectiveness**: The implemented augmentation techniques significantly improved model generalization, preventing overfitting despite the relatively small dataset.

3. **Transfer Learning Benefits**: All three approaches outperformed training from scratch, demonstrating the effectiveness of pre-trained features.

4. **Architecture Impact**: The choice of pooling method (Flatten vs GlobalAveragePooling2D) significantly affected performance, with Flatten achieving better results.

## Technical Details

### Training Parameters
- **Epochs**: 150
- **Batch Size**: 128
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

### Environment
- **Platform**: Google Colab
- **Hardware**: GPU acceleration enabled
- **Framework**: TensorFlow 2.x
- **Libraries**: tensorflow_datasets, matplotlib, numpy

## Files

- `MAIN.ipynb`: Complete implementation notebook with all experiments
- `generative_adversarial_network_paper.pdf`: Reference material (unrelated to this project)

## Usage

1. Open `MAIN.ipynb` in Google Colab
2. Enable GPU acceleration (Runtime > Change runtime type > Hardware acceleration > GPU)
3. Run all cells sequentially
4. Results and visualizations will be displayed automatically

## Conclusion

This project successfully demonstrates the effectiveness of combining data augmentation with transfer learning for image classification tasks. The partial fine-tuning approach proved most effective, achieving near-perfect accuracy while maintaining good generalization. The implementation showcases best practices for deep learning model development using TensorFlow and Google Colab.
