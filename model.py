import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
INPUT_SHAPE = (224, 224, 3)  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 2  # Number of disease classes in our dataset

# Class names matching the app's disease IDs
# TE
CLASS_NAMES = [
    'Algal leaf spot',
    'leaf blight'
]

# Paths
#TSEJS
BASE_PATH = 'dataset'
MODEL_PATH = 'models/disease_detection_model.h5'
TF_LITE_MODEL_PATH = 'models/disease_detection_model.tflite'
TF_LITE_QUANTIZED_PATH = 'models/disease_detection_model_quantized.tflite'

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def create_data_generators():
    """Create training and validation data generators with augmentation"""
    
    # Data augmentation for training the data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    #hgi muyo unotan  unotan unoys unst
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        BASE_PATH,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        BASE_PATH,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_model():
    """Create MobileNetV2 model with custom classification head"""
    
    # Load MobileNetV2 with pre-trained weights, excluding top layers
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks for early stopping and learning rate reduction"""
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some of the base model layers"""
    
    # Unfreeze the last 23 layers (last 3 blocks of MobileNetV2)
    for layer in base_model.layers[-23:]:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune with the same callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Fine-tune the model
    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=20,  # Fewer epochs for fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine_tune

def evaluate_model(model, validation_generator):
    """Evaluate the model and generate classification report and confusion matrix"""
    
    # Get the true labels
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Get predictions
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES
    ))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate overall metrics
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Calculate per-class metrics
    class_accuracies = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            class_acc = np.sum(y_pred[class_indices] == i) / len(class_indices)
            class_accuracies[class_name] = class_acc
            print(f"Accuracy for {class_name}: {class_acc:.4f}")
    
    return accuracy, class_accuracies

def convert_to_tflite(model):
    """Convert the trained model to TensorFlow Lite format"""
    
    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    with open(TF_LITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to {TF_LITE_MODEL_PATH}")
    
    # Convert to quantized model for better mobile performance
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    
    # Save the quantized model
    with open(TF_LITE_QUANTIZED_PATH, 'wb') as f:
        f.write(quantized_tflite_model)
    
    print(f"Quantized TensorFlow Lite model saved to {TF_LITE_QUANTIZED_PATH}")
    
    # Calculate size reduction
    original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TF_LITE_MODEL_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(TF_LITE_QUANTIZED_PATH) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    print(f"Quantized TFLite model size: {quantized_size:.2f} MB")
    
    return tflite_model, quantized_tflite_model

def test_tflite_model(tflite_model, validation_generator):
    """Test the TensorFlow Lite model on a few sample images"""
    
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on a few samples
    validation_generator.reset()
    batch_x, batch_y = next(validation_generator)
    
    # Test on 5 samples
    for i in range(5):
        # Get sample
        input_data = np.expand_dims(batch_x[i], axis=0).astype(np.float32)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class
        tflite_prediction = np.argmax(tflite_output[0])
        true_class = np.argmax(batch_y[i])
        
        print(f"Sample {i+1}:")
        print(f"  True class: {CLASS_NAMES[true_class]}")
        print(f"  Predicted class: {CLASS_NAMES[tflite_prediction]}")
        print(f"  Confidence: {tflite_output[0][tflite_prediction]:.4f}")
        print()

def plot_training_history(history):
    """Plot training and validation accuracy and loss"""
    
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """Main function to run the training pipeline"""
    
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators()
    
    print("Creating model...")
    model, base_model = create_model()
    model.summary()
    
    print("Training model (phase 1)...")
    history = train_model(model, train_generator, validation_generator)
    
    print("Fine-tuning model (phase 2)...")
    history_fine_tune = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    print("Evaluating model...")
    accuracy, class_accuracies = evaluate_model(model, validation_generator)
    
    print("Converting model to TensorFlow Lite...")
    tflite_model, quantized_tflite_model = convert_to_tflite(model)
    
    print("Testing TensorFlow Lite model...")
    test_tflite_model(quantized_tflite_model, validation_generator)
    
    print("Plotting training history...")
    # Combine histories
    combined_history = {}
    for key in history.history:
        combined_history[key] = history.history[key] + history_fine_tune.history[key]
    
    # Create a mock history object
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    plot_training_history(CombinedHistory(combined_history))
    
    print("Training complete!")
    print(f"Final model saved to {MODEL_PATH}")
    print(f"TensorFlow Lite model saved to {TF_LITE_MODEL_PATH}")
    print(f"Quantized TensorFlow Lite model saved to {TF_LITE_QUANTIZED_PATH}")

if __name__ == "__main__":
    main()