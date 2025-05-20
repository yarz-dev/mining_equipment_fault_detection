import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class AcousticModel:
    def __init__(self, input_shape=(128, 128, 1), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.label_encoder = LabelEncoder()
        
    def _build_model(self):
        """Build the CNN model for acoustic data classification"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_audio(self, audio_data, sr=22050):
        """Convert raw audio to mel spectrogram"""
        # Ensure input is float32
        audio_data = np.array(audio_data, dtype=np.float32)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Reshape for model input
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)
        
        return mel_spec_norm.astype(np.float32)
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        """Train the model with early stopping"""
        # Ensure data types
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int32)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Ensure data types
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        return test_loss, test_acc
    
    def predict(self, audio_data):
        """Make predictions on new audio data"""
        # Preprocess the audio
        mel_spec = self.preprocess_audio(audio_data)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        # Make prediction
        predictions = self.model.predict(mel_spec)
        return predictions
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = models.load_model(path) 