"""
Hybrid SVM-RNN Fake Facebook Account Detection System
Uses C-SVM with RBF Kernel for CSV features and Bidirectional LSTM for NLP features
Handles Personal Accounts and Pages with different feature sets
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class HybridFacebookFakeAccountDetector:
    def __init__(self):
        self.svm_model = None
        self.rnn_model = None
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_length = 100
        self.all_csv_features = None
        self.vocab_size = 5000
        
        # Define feature sets for different account types
        # Features to EXCLUDE for Personal accounts (personal(0)/page(1) = 0)
        self.personal_exclude_features = ['#followers', '#following', 'category']
        
        # Features to EXCLUDE for Pages (personal(0)/page(1) = 1)
        self.page_exclude_features = ['private', '#friends', 'friends visibility']
        
    def load_data(self, csv_path='CSV.csv', json_path='NLP.json'):
        """Load CSV and JSON data"""
        print("Loading data...")
        
        # Load CSV data
        csv_data = pd.read_csv(csv_path)
        
        # Remove empty rows
        csv_data = csv_data.dropna(how='all')
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Extract all features from CSV (excluding 'fake' column)
        self.all_csv_features = csv_data.columns.tolist()
        if 'fake' in self.all_csv_features:
            self.all_csv_features.remove('fake')
        
        # Get account types and labels
        account_types = csv_data['personal(0)/page(1)'].values
        y = csv_data['fake'].values
        
        # Process features based on account type
        X_csv_processed = []
        
        for idx in range(len(csv_data)):
            account_type = account_types[idx]
            row_data = csv_data.iloc[idx]
            
            # Select features based on account type
            if account_type == 0:  # Personal account
                # Use all features except those in personal_exclude_features
                features_to_use = [f for f in self.all_csv_features 
                                 if f not in self.personal_exclude_features]
            else:  # Page (account_type == 1)
                # Use all features except those in page_exclude_features
                features_to_use = [f for f in self.all_csv_features 
                                 if f not in self.page_exclude_features]
            
            # Extract values for selected features
            feature_values = row_data[features_to_use].values
            X_csv_processed.append(feature_values)
        
        X_csv = np.array(X_csv_processed)
        
        # Extract text features from JSON
        X_nlp = []
        for entry in json_data:
            # Combine all text fields (fullname, bio, work, education, categories, captions, comments)
            text = f"{entry.get('fullname', '')} {entry.get('bio', '')} "
            text += f"{entry.get('work', '')} {entry.get('education', '')} {entry.get('categories', '')}"
            text += ' '.join(entry.get('captions', []))
            text += ' '.join(entry.get('comments', []))
            X_nlp.append(text)
        
        print(f"Loaded {len(X_csv)} samples")
        print(f"Personal accounts: {np.sum(account_types == 0)}")
        print(f"Page accounts: {np.sum(account_types == 1)}")
        
        return X_csv, X_nlp, y, account_types
    
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
    
    def train_svm(self, X_train, y_train):
        """Train C-SVM with RBF Kernel"""
        print("\nTraining SVM model...")
        self.svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        self.svm_model.fit(X_train, y_train)
        print("SVM training completed!")
    
    def train_rnn(self, X_train, y_train, X_val, y_val):
        """Train Bidirectional LSTM RNN"""
        print("\nTraining RNN model...")
        
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
        
        print("RNN training completed!")
        return history
    
    def train(self, csv_path='CSV.csv', json_path='NLP.json', test_size=0.2):
        """Complete training pipeline"""
        # Load data
        X_csv, X_nlp, y, account_types = self.load_data(csv_path, json_path)
        
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
        
        account_types_test = account_types[test_idx]
        
        # Train models
        self.train_svm(X_csv_train, y_train)
        self.train_rnn(X_nlp_train, y_train, X_nlp_val, y_val)
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        self.evaluate(X_csv_test, X_nlp_test, y_test, account_types_test)
        
        # Save models
        self.save_models()
        
        return X_csv_test, X_nlp_test, y_test, account_types_test
    
    def predict(self, X_csv, X_nlp, return_confidence=True):
        """Hybrid prediction combining SVM and RNN"""
        # Get SVM predictions
        svm_proba = self.svm_model.predict_proba(X_csv)[:, 1]
        
        # Get RNN predictions
        rnn_proba = self.rnn_model.predict(X_nlp, verbose=0).flatten()
        
        # Combine predictions (weighted average)
        # SVM: 40%, RNN: 60% (NLP features often more informative for social media)
        combined_proba = 0.4 * svm_proba + 0.6 * rnn_proba
        
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
    
    def evaluate(self, X_csv_test, X_nlp_test, y_test, account_types_test):
        """Evaluate model performance"""
        predictions, confidence = self.predict(X_csv_test, X_nlp_test)
        
        # Calculate metrics
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"\nOverall Performance:")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Separate evaluation for Personal and Page accounts
        personal_mask = account_types_test == 0
        page_mask = account_types_test == 1
        
        if np.sum(personal_mask) > 0:
            print("\n" + "-"*60)
            print("Performance on PERSONAL ACCOUNTS:")
            personal_acc = accuracy_score(y_test[personal_mask], predictions[personal_mask])
            print(f"Accuracy: {personal_acc:.4f} ({personal_acc*100:.2f}%)")
            print(f"Samples: {np.sum(personal_mask)}")
        
        if np.sum(page_mask) > 0:
            print("\n" + "-"*60)
            print("Performance on PAGE ACCOUNTS:")
            page_acc = accuracy_score(y_test[page_mask], predictions[page_mask])
            print(f"Accuracy: {page_acc:.4f} ({page_acc*100:.2f}%)")
            print(f"Samples: {np.sum(page_mask)}")
        
        # Show sample predictions with confidence
        print("\n" + "-"*60)
        print("Sample Predictions:")
        print("-" * 60)
        for i in range(min(10, len(predictions))):
            result = "Fake" if predictions[i] == 1 else "Real"
            actual = "Fake" if y_test[i] == 1 else "Real"
            acc_type = "Personal" if account_types_test[i] == 0 else "Page"
            status = "✓" if predictions[i] == y_test[i] else "✗"
            print(f"{status} {result} ({confidence[i]:.1f}% confidence) - Actual: {actual} [{acc_type}]")
        
        # --- ROC Curve ---
        # Get combined probability scores (same logic as predict)
        svm_proba = self.svm_model.predict_proba(X_csv_test)[:, 1]
        rnn_proba = self.rnn_model.predict(X_nlp_test, verbose=0).flatten()
        combined_proba = 0.4 * svm_proba + 0.6 * rnn_proba

        # Compute ROC curves for Hybrid, SVM-only, and RNN-only
        fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, combined_proba)
        auc_hybrid = auc(fpr_hybrid, tpr_hybrid)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_hybrid, tpr_hybrid, color='darkorange', lw=2,
                 label=f'Hybrid SVM-RNN (AUC = {auc_hybrid:.4f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve – Fake Facebook Account Detection', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150)
        plt.show()
        print("\nROC curve saved as 'roc_curve.png'")

        return precision, recall, accuracy
    
    def predict_single(self, csv_features_dict, text_data, account_type):
        """
        Predict for a single Facebook account
        
        Parameters:
        -----------
        csv_features_dict : dict
            Dictionary containing all CSV features
        text_data : str
            Combined text from fullname, bio, work, education, categories, captions, comments
        account_type : int
            0 for Personal account, 1 for Page
        """
        # Select appropriate features based on account type
        if account_type == 0:  # Personal account
            features_to_use = [f for f in self.all_csv_features 
                             if f not in self.personal_exclude_features]
        else:  # Page
            features_to_use = [f for f in self.all_csv_features 
                             if f not in self.page_exclude_features]
        
        # Extract values for selected features
        csv_array = np.array([csv_features_dict[f] for f in features_to_use]).reshape(1, -1)
        csv_scaled = self.scaler.transform(csv_array)
        
        # Preprocess text
        nlp_processed = self.preprocess_nlp([text_data], fit=False)
        
        # Get prediction
        prediction, confidence = self.predict(csv_scaled, nlp_processed)
        
        result = "Fake" if prediction[0] == 1 else "Real"
        conf = confidence[0]
        acc_type_str = "Personal" if account_type == 0 else "Page"
        
        return result, conf, acc_type_str
    
    def save_models(self, svm_path='fb_svm_model.pkl', rnn_path='fb_rnn_model.h5', 
                    scaler_path='fb_scaler.pkl', tokenizer_path='fb_tokenizer.pkl',
                    metadata_path='fb_model_metadata.pkl'):
        """Save trained models and metadata"""
        print("\nSaving models...")
        
        # Save SVM
        with open(svm_path, 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        # Save RNN
        self.rnn_model.save(rnn_path)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save metadata (feature lists)
        metadata = {
            'all_csv_features': self.all_csv_features,
            'personal_exclude_features': self.personal_exclude_features,
            'page_exclude_features': self.page_exclude_features
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Models saved successfully!")
        print(f"  - SVM: {svm_path}")
        print(f"  - RNN: {rnn_path}")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Tokenizer: {tokenizer_path}")
        print(f"  - Metadata: {metadata_path}")
    
    def load_models(self, svm_path='fb_svm_model.pkl', rnn_path='fb_rnn_model.h5',
                    scaler_path='fb_scaler.pkl', tokenizer_path='fb_tokenizer.pkl',
                    metadata_path='fb_model_metadata.pkl'):
        """Load trained models and metadata"""
        print("Loading models...")
        
        # Load SVM
        with open(svm_path, 'rb') as f:
            self.svm_model = pickle.load(f)
        
        # Load RNN
        self.rnn_model = load_model(rnn_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.all_csv_features = metadata['all_csv_features']
            self.personal_exclude_features = metadata['personal_exclude_features']
            self.page_exclude_features = metadata['page_exclude_features']
        
        print("Models loaded successfully!")


# Training script
if __name__ == "__main__":
    print("="*60)
    print("HYBRID SVM-RNN FAKE FACEBOOK ACCOUNT DETECTION SYSTEM")
    print("="*60)
    print("\nFeature Selection Rules:")
    print("  Personal Accounts: Exclude #followers, #following, category")
    print("  Page Accounts: Exclude private, #friends, friends visibility")
    print("="*60)
    
    # Initialize detector
    detector = HybridFacebookFakeAccountDetector()
    
    # Train the model
    try:
        detector.train('CSV.csv', 'NLP.json', test_size=0.2)
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please ensure CSV.csv and NLP.json are in the same directory.")