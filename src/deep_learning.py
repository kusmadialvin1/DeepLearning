import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt

def create_resnet_model(num_classes, input_shape=(224, 224, 3), freeze_base=True):
    """
    Create a ResNet50 model with transfer learning.

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        freeze_base: Whether to freeze the base model layers

    Returns:
        model: Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    if freeze_base:
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def create_vgg_model(num_classes, input_shape=(224, 224, 3), freeze_base=True):
    """
    Create a VGG16 model with transfer learning.

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        freeze_base: Whether to freeze the base model layers

    Returns:
        model: Compiled Keras model
    """
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if freeze_base:
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer

    Returns:
        model: Compiled model
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_generator, val_generator, epochs=50, model_name='resnet'):
    """
    Train the model with early stopping and model checkpointing.

    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Maximum number of epochs
        model_name: Name for saving the model

    Returns:
        history: Training history
    """
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'models/{model_name}_best.h5',
                                monitor='val_accuracy',
                                save_best_only=True,
                                mode='max')

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint]
    )

    return history

def fine_tune_model(model, train_generator, val_generator, learning_rate=1e-5, epochs=20):
    """
    Fine-tune the model by unfreezing some layers.

    Args:
        model: Trained model
        train_generator: Training data generator
        val_generator: Validation data generator
        learning_rate: Lower learning rate for fine-tuning
        epochs: Number of fine-tuning epochs

    Returns:
        history: Fine-tuning history
    """
    # Unfreeze the last few layers of the base model
    for layer in model.layers[-20:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=learning_rate)

    # Fine-tune
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    return history

def plot_training_history(history, model_name='Model'):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Training history object
        model_name: Name of the model for plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower()}_training_history.png')
    plt.show()

def save_model(model, model_name='resnet'):
    """
    Save the trained model.

    Args:
        model: Trained Keras model
        model_name: Name for the saved model file
    """
    model.save(f'models/{model_name}_final.h5')
    print(f"Model saved as models/{model_name}_final.h5")

if __name__ == "__main__":
    print("Deep learning module loaded. Use functions to create and train CNN models.")
