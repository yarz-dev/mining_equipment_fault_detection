import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from acoustic_model import AcousticModel
import matplotlib.pyplot as plt
from collections import Counter
import librosa

def load_data(data_dir):
    """Load processed .npy files from the data directory and convert to mel spectrograms"""
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))
    
    print(f"\nLoading data from: {data_dir}")
    print(f"Found classes: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        print(f"\nProcessing class {class_name}:")
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load raw audio data
                    audio_data = np.load(file_path)
                    
                    # Convert to mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio_data,
                        sr=22050,  # Assuming 22.05kHz sampling rate
                        n_mels=128,
                        fmax=8000,
                        n_fft=2048,
                        hop_length=512
                    )
                    
                    # Convert to log scale
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalize
                    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                    
                    # Resize to 128x128 if needed
                    if mel_spec_norm.shape[1] != 128:
                        mel_spec_norm = librosa.util.fix_length(mel_spec_norm, size=128, axis=1)
                    
                    # Add channel dimension
                    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)
                    
                    X.append(mel_spec_norm)
                    y.append(class_idx)
                    print(f"Successfully processed {file_name}")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    continue
    
    if not X:
        raise ValueError("No valid data files found!")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nLoaded data summary:")
    print(f"Number of samples: {len(X)}")
    print(f"Data shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Label type: {y.dtype}")
    
    return X, y, class_names

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load data
    train_dir = 'datasets/cleaned/acoustic/train'
    test_dir = 'datasets/cleaned/acoustic/test'
    
    print("Loading training data...")
    X_train, y_train, class_names = load_data(train_dir)
    print("Loading test data...")
    X_test, y_test, _ = load_data(test_dir)
    
    # Print class distribution
    train_dist = Counter(y_train)
    print("\nTraining data class distribution:")
    for class_idx, count in train_dist.items():
        print(f"Class {class_names[class_idx]}: {count} samples")
    
    # Check if we have enough samples for stratified split
    min_samples = min(train_dist.values())
    if min_samples < 2:
        print("\nWarning: Some classes have too few samples for stratified split.")
        print("Using non-stratified split instead.")
        stratify = None
    else:
        stratify = y_train
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.15,  # 15% for validation
        random_state=42,  # For reproducibility
        stratify=stratify  # Will be None if we have too few samples
    )
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print validation set distribution
    val_dist = Counter(y_val)
    print("\nValidation set class distribution:")
    for class_idx, count in val_dist.items():
        print(f"Class {class_names[class_idx]}: {count} samples")
    
    # Initialize and train model
    model = AcousticModel(input_shape=(128, 128, 1), num_classes=len(class_names))
    
    print("\nTraining model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model.save_model('models/acoustic_model.h5')
    print("\nModel saved to 'models/acoustic_model.h5'")

if __name__ == "__main__":
    main() 