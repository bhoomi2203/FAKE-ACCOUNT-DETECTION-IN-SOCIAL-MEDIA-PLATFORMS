"""
Hybrid Balanced Random Forest + Bi-LSTM RNN Fake Instagram Account Detection System
Uses Balanced Random Forest for CSV features and Bidirectional LSTM for NLP features
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class HybridBRFRNNDetector:
    def __init__(self):
        self.brf_model = None
        self.rnn_model = None
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_length = 100
        self.csv_features = None
        self.vocab_size = 5000
        
    def load_data(self, csv_path='train_csv.csv', json_path='train_nlp.json'):
        """Load CSV and JSON data"""
        print("Loading data...")
        
        # Load CSV data
        csv_data = pd.read_csv(csv_path)
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract features from CSV (excluding 'fake' column)
        self.csv_features = csv_data.columns.tolist()
        if 'fake' in self.csv_features:
            self.csv_features.remove('fake')
        
        X_csv = csv_data[self.csv_features].values
        y = csv_data['fake'].values
        
        # Extract text features from JSON
        X_nlp = []
        for entry in json_data:
            # Combine all text fields
            text = f"{entry.get('username', '')} {entry.get('fullname', '')} {entry.get('bio', '')} "
            text += ' '.join(entry.get('captions', []))
            text += ' '.join(entry.get('comments', []))
            X_nlp.append(text)
        
        print(f"Loaded {len(X_csv)} samples")
        return X_csv, X_nlp, y
    
    def preprocess_csv(self, X_csv, fit=True):
        """Preprocess CSV numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X_csv)
        else:
            X_scaled = self.scaler.transform(X_csv)
        return X_scaled
    
    def preprocess_nlp(self, X_nlp, fit=True):
        """Preprocess NLP text features"""
        if fit:
            self.tokenizer.fit_on_texts(X_nlp)
        
        sequences = self.tokenizer.texts_to_sequences(X_nlp)
        X_padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return X_padded
    
    def train_brf(self, X_train, y_train):
        """Train Balanced Random Forest"""
        print("\nTraining Balanced Random Forest model...")
        self.brf_model = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            sampling_strategy='auto',  # Automatically balance classes
            replacement=True  # Bootstrap sampling with replacement
        )
        self.brf_model.fit(X_train, y_train)
        print("Balanced Random Forest training completed!")
    
    def train_rnn(self, X_train, y_train, X_val, y_val):
        """Train Bidirectional LSTM RNN"""
        print("\nTraining Bi-LSTM RNN model...")
        
        self.rnn_model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        self.rnn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.rnn_model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("Bi-LSTM RNN training completed!")
        return history
    
    def train(self, csv_path='train_csv.csv', json_path='train_nlp.json', test_size=0.2):
        """Complete training pipeline"""
        # Load data
        X_csv, X_nlp, y = self.load_data(csv_path, json_path)
        
        # Split data (80-20)
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )
        
        # Further split training data for validation
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=42, stratify=y[train_idx]
        )
        
        # Prepare CSV data
        X_csv_train = self.preprocess_csv(X_csv[train_idx], fit=True)
        X_csv_test = self.preprocess_csv(X_csv[test_idx], fit=False)
        
        # Prepare NLP data
        X_nlp_list = [X_nlp[i] for i in range(len(X_nlp))]
        X_nlp_train = self.preprocess_nlp([X_nlp_list[i] for i in train_idx], fit=True)
        X_nlp_val = self.preprocess_nlp([X_nlp_list[i] for i in val_idx], fit=False)
        X_nlp_test = self.preprocess_nlp([X_nlp_list[i] for i in test_idx], fit=False)
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        # Train models
        self.train_brf(X_csv_train, y_train)
        self.train_rnn(X_nlp_train, y_train, X_nlp_val, y_val)
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        self.evaluate(X_csv_test, X_nlp_test, y_test)
        
        # Save models
        self.save_models()
        
        return X_csv_test, X_nlp_test, y_test
    
    def predict(self, X_csv, X_nlp, return_confidence=True):
        """Hybrid prediction combining BRF and RNN"""
        # Get BRF predictions (probability)
        brf_proba = self.brf_model.predict_proba(X_csv)[:, 1]
        
        # Get RNN predictions
        rnn_proba = self.rnn_model.predict(X_nlp, verbose=0).flatten()
        
        # Combine predictions (weighted average)
        # BRF: 40%, RNN: 60% (NLP features often more informative for social media)
        combined_proba = 0.4 * brf_proba + 0.6 * rnn_proba
        
        predictions = (combined_proba >= 0.5).astype(int)
        
        if return_confidence:
            # Convert to confidence percentage
            confidence = np.where(
                predictions == 1,
                combined_proba * 100,
                (1 - combined_proba) * 100
            )
            return predictions, confidence
        
        return predictions
    
    def evaluate(self, X_csv_test, X_nlp_test, y_test):
        """Evaluate model performance"""
        predictions, confidence = self.predict(X_csv_test, X_nlp_test)
        
        # Calculate metrics
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Show sample predictions with confidence
        print("\nSample Predictions:")
        print("-" * 40)
        for i in range(min(10, len(predictions))):
            result = "Fake" if predictions[i] == 1 else "Real"
            actual = "Fake" if y_test[i] == 1 else "Real"
            status = "✓" if predictions[i] == y_test[i] else "✗"
            print(f"{status} {result} ({confidence[i]:.1f}% confidence) - Actual: {actual}")
        
        return precision, recall, accuracy
    
    def predict_single(self, csv_features, text_data):
        """Predict for a single account"""
        # Preprocess CSV features
        csv_array = np.array(csv_features).reshape(1, -1)
        csv_scaled = self.scaler.transform(csv_array)
        
        # Preprocess text
        nlp_processed = self.preprocess_nlp([text_data], fit=False)
        
        # Get prediction
        prediction, confidence = self.predict(csv_scaled, nlp_processed)
        
        result = "Fake" if prediction[0] == 1 else "Real"
        conf = confidence[0]
        
        return result, conf
    
    def save_models(self, brf_path='brf_model.pkl', rnn_path='rnn_model.h5', 
                    scaler_path='scaler.pkl', tokenizer_path='tokenizer.pkl'):
        """Save trained models"""
        print("\nSaving models...")
        
        # Save BRF
        with open(brf_path, 'wb') as f:
            pickle.dump(self.brf_model, f)
        
        # Save RNN
        self.rnn_model.save(rnn_path)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print("Models saved successfully!")
    
    def load_models(self, brf_path='brf_model.pkl', rnn_path='rnn_model.h5',
                    scaler_path='scaler.pkl', tokenizer_path='tokenizer.pkl'):
        """Load trained models"""
        print("Loading models...")
        
        # Load BRF
        with open(brf_path, 'rb') as f:
            self.brf_model = pickle.load(f)
        
        # Load RNN
        self.rnn_model = load_model(rnn_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print("Models loaded successfully!")

# Training script
if __name__ == "__main__":
    print("="*60)
    print("HYBRID BALANCED RANDOM FOREST + BI-LSTM RNN")
    print("FAKE ACCOUNT DETECTION SYSTEM")
    print("="*60)
    
    # Initialize detector
    detector = HybridBRFRNNDetector()
    
    # Train the model
    try:
        detector.train('train_csv.csv', 'train_nlp.json', test_size=0.2)
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please ensure train_csv.csv and train_nlp.json are in the same directory.")