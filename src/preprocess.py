import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import shutil

def load_train_data(train_path='TRAIN', img_size=(224, 224)):
    """
    Load training data from TRAIN folder.

    Args:
        train_path: Path to the TRAIN directory
        img_size: Target image size for resizing

    Returns:
        images: List of preprocessed images
        labels: List of corresponding labels
        class_names: List of class names
    """
    images = []
    labels = []
    class_names = []

    # Get all class directories
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Train path {train_path} does not exist.")

    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]

    for class_idx, class_dir in enumerate(sorted(class_dirs)):
        class_name = class_dir.name
        class_names.append(class_name)

        print(f"Loading train class: {class_name}")

        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} train images from {len(class_names)} classes")
    return images, labels, class_names

def load_test_data(test_path='TEST', img_size=(224, 224)):
    """
    Load test data from TEST folder.

    Args:
        test_path: Path to the TEST directory
        img_size: Target image size for resizing

    Returns:
        images: List of preprocessed images
        labels: List of corresponding labels
        class_names: List of class names
    """
    images = []
    labels = []
    class_names = []

    # Get all class directories
    test_path = Path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test path {test_path} does not exist.")

    class_dirs = [d for d in test_path.iterdir() if d.is_dir()]

    for class_idx, class_dir in enumerate(sorted(class_dirs)):
        class_name = class_dir.name
        class_names.append(class_name)

        print(f"Loading test class: {class_name}")

        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} test images from {len(class_names)} classes")
    return images, labels, class_names

def create_train_val_test_split(images, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.

    Args:
        images: Array of images
        labels: Array of labels
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random state for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train + val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, stratify=y_train_val, random_state=random_state
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32, augmentation=True):
    """
    Create data generators for training and validation with optional augmentation.

    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size for generators
        augmentation: Whether to apply data augmentation

    Returns:
        train_generator, val_generator
    """
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    return train_generator, val_generator

def save_test_data(X_test, y_test, class_names, save_path='data/processed'):
    """
    Save test data for later evaluation.

    Args:
        X_test: Test images
        y_test: Test labels
        class_names: List of class names
        save_path: Path to save processed data
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(save_path, 'X_test.npy'), X_test)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)
    np.save(os.path.join(save_path, 'class_names.npy'), class_names)

    print(f"Test data saved to {save_path}")

if __name__ == "__main__":
    # Load train data
    X_train_full, y_train_full, class_names = load_train_data()

    # Split train data into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    # Load test data
    X_test, y_test, _ = load_test_data()

    # Save test data
    save_test_data(X_test, y_test, class_names)
