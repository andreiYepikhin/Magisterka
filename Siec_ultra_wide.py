import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class ADHDClassifier:
    def __init__(self, data_folder='data', model_save_folder='saved_models'):
        self.data_folder = data_folder
        self.model_save_folder = model_save_folder
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
     
        os.makedirs(model_save_folder, exist_ok=True)
        
    def load_data(self):
        features_list = []
        labels_list = []
        
        print(f"Loading data from {self.data_folder}...")

        pkl_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {self.data_folder}")
        
        print(f"Found {len(pkl_files)} pickle files")
        
        for file in pkl_files:
            file_path = os.path.join(self.data_folder, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    
                features = data['features']
                label = data['adhd_label']
                
                features_list.append(features)
                labels_list.append(label)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
        print(f"Class distribution - ADHD: {np.sum(y)}, Non-ADHD: {len(y) - np.sum(y)}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 
                   'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'Pz', 'T5', 'T6']

        channel_features = []
        
        for channel_idx in range(19): 

            start_idx = channel_idx * 5
            end_idx = start_idx + 5
            channel_data = X_scaled[:, start_idx:end_idx]  
            channel_features.append(channel_data)
        
        print(f"Created {len(channel_features)} channel branches")
        for i, channel_data in enumerate(channel_features):
            print(f"Channel {channels[i]} features shape: {channel_data.shape}")

        channel_inputs = []
        for channel_data in channel_features:
            channel_input = channel_data.reshape(channel_data.shape[0], channel_data.shape[1], 1)
            channel_inputs.append(channel_input)
        
        return channel_inputs, y
    
    def create_balanced_test_set(self, X, y, test_size=0.03):

        adhd_indices = np.where(y == 1)[0]
        non_adhd_indices = np.where(y == 0)[0]

        total_test_size = int(len(X) * test_size)
        test_size_per_class = total_test_size // 2

        np.random.seed(42)
        test_adhd_indices = np.random.choice(adhd_indices, test_size_per_class, replace=False)
        test_non_adhd_indices = np.random.choice(non_adhd_indices, test_size_per_class, replace=False)
        
        test_indices = np.concatenate([test_adhd_indices, test_non_adhd_indices])
        train_val_indices = np.setdiff1d(np.arange(len(X)), test_indices)

        X_train_val, X_test = X[train_val_indices], X[test_indices]
        y_train_val, y_test = y[train_val_indices], y[test_indices]
        
        print(f"Test set: {len(X_test)} samples (ADHD: {np.sum(y_test)}, Non-ADHD: {len(y_test) - np.sum(y_test)})")
        print(f"Train+Val set: {len(X_train_val)} samples")
        
        return X_train_val, X_test, y_train_val, y_test
    
    def build_cnn_model(self, input_shapes):

        channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 
                   'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'Pz', 'T5', 'T6']
        
        channel_inputs = []
        for i, channel in enumerate(channels):
            channel_input = Input(shape=input_shapes[i], name=f'{channel}_input')
            channel_inputs.append(channel_input)

        def create_channel_branch(input_layer, channel_name):
            x = Conv1D(16, 3, activation='relu', name=f'{channel_name}_conv1')(input_layer)
            x = BatchNormalization(name=f'{channel_name}_bn1')(x)
            x = Dropout(0.2, name=f'{channel_name}_dropout1')(x)
            
            x = Conv1D(32, 2, activation='relu', name=f'{channel_name}_conv2')(x)
            x = BatchNormalization(name=f'{channel_name}_bn2')(x)
            x = Dropout(0.2, name=f'{channel_name}_dropout2')(x)
            
            x = Flatten(name=f'{channel_name}_flatten')(x)
            x = Dense(64, activation='relu', name=f'{channel_name}_dense')(x)
            x = BatchNormalization(name=f'{channel_name}_bn3')(x)
            x = Dropout(0.3, name=f'{channel_name}_dropout3')(x)
            
            return x

        channel_branches = []
        for i, channel in enumerate(channels):
            branch = create_channel_branch(channel_inputs[i], channel)
            channel_branches.append(branch)

        combined = concatenate(channel_branches, name='concatenate_all_channels')

        x = Dense(512, activation='relu', name='final_dense1')(combined)
        x = BatchNormalization(name='final_bn1')(x)
        x = Dropout(0.5, name='final_dropout1')(x)
        
        x = Dense(256, activation='relu', name='final_dense2')(x)
        x = BatchNormalization(name='final_bn2')(x)
        x = Dropout(0.5, name='final_dropout2')(x)
        
        x = Dense(128, activation='relu', name='final_dense3')(x)
        x = BatchNormalization(name='final_bn3')(x)
        x = Dropout(0.3, name='final_dropout3')(x)
        
        output = Dense(1, activation='sigmoid', name='output')(x)

        model = Model(inputs=channel_inputs, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        input_shapes = [x.shape[1:] for x in X_train]  
        self.model = self.build_cnn_model(input_shapes)
        
        print("Model Architecture:")
        self.model.summary()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.0001
        )
      
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=150,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def plot_learning_curves(self):

        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_folder, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validate(self, X, y, k=5):
        print(f"\nPerforming {k}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X[0], y), 1):
            print(f"Training fold {fold}/{k}...")

            X_train_fold = [x[train_idx] for x in X]
            X_val_fold = [x[val_idx] for x in X]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            input_shapes = [x.shape[1:] for x in X_train_fold]
            fold_model = self.build_cnn_model(input_shapes)

            fold_model.fit(
                X_train_fold, y_train_fold,
                batch_size=32,
                epochs=50,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0
            )

            val_loss, val_acc = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(val_acc)
            print(f"Fold {fold} validation accuracy: {val_acc:.4f}")
        
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        return cv_scores
    
    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            print("No trained model available. Train the model first.")
            return

        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-ADHD', 'ADHD']))

        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-ADHD', 'ADHD'],
                   yticklabels=['Non-ADHD', 'ADHD'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.model_save_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, cm
    
    def save_model(self):
        if self.model is None:
            print("No trained model to save.")
            return

        model_path = os.path.join(self.model_save_folder, 'adhd_cnn_channel_model.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        scaler_path = os.path.join(self.model_save_folder, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")

def main():
    classifier = ADHDClassifier(data_folder='Przetwarzanie_bazy_danych/FurryV2', 
                               model_save_folder='UltraWideV2')
    
    try:
        X, y = classifier.load_data()
        X_train_val, X_test, y_train_val, y_test = classifier.create_balanced_test_set(X, y)

        X_train_val_processed, y_train_val = classifier.preprocess_data(X_train_val, y_train_val)
        X_test_processed, y_test = classifier.preprocess_data(X_test, y_test)

        train_indices, val_indices = train_test_split(
            np.arange(len(y_train_val)), 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train_val
        )
        
        X_train = [x[train_indices] for x in X_train_val_processed]
        X_val = [x[val_indices] for x in X_train_val_processed]
        y_train = y_train_val[train_indices]
        y_val = y_train_val[val_indices]
        
        print(f"\nFinal data splits:")
        print(f"Training: {len(X_train[0])} samples across {len(X_train)} channels")
        print(f"Validation: {len(X_val[0])} samples across {len(X_val)} channels") 
        print(f"Test: {len(X_test_processed[0])} samples across {len(X_test_processed)} channels")

        classifier.train_model(X_train, y_train, X_val, y_val)
        classifier.plot_learning_curves()
        classifier.cross_validate(X_train_val_processed, y_train_val, k=5)
        classifier.evaluate_model(X_test_processed, y_test)
        classifier.save_model()
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
