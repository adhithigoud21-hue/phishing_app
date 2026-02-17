import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedPhishingDetector:
    def __init__(self, data_path='PhiUSIIL_Phishing_URL_Dataset.csv', min_samples=30000):
        self.data_path = data_path
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("=" * 70)
        print("📊 LOADING PHISHING DETECTION DATASET")
        print("=" * 70)
        
        # Load dataset
        df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Use at least min_samples rows
        if len(df) > self.min_samples:
            # Shuffle and sample
            df = df.sample(n=min(len(df), self.min_samples * 2), random_state=42)
            print(f"📦 Using {len(df):,} rows for training")
        
        # Check for label column
        label_column = None
        possible_labels = ['label', 'Label', 'class', 'Class', 'target', 'Target', 'phishing', 'Phishing']
        
        for col in possible_labels:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            # Assume last column is the label
            label_column = df.columns[-1]
            print(f"⚠️  Assuming '{label_column}' is the label column")
        else:
            print(f"✅ Found label column: '{label_column}'")
        
        # Separate features and labels
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        # Remove text columns that cannot be used as features
        text_columns = ['URL', 'Domain', 'TLD', 'Title']
        existing_text_cols = [col for col in text_columns if col in X.columns]
        if existing_text_cols:
            X = X.drop(columns=existing_text_cols)
            print(f"✅ Removed text columns: {existing_text_cols}")
        
        # Convert labels to binary (0: legitimate, 1: phishing)
        if y.dtype == 'object':
            # Map string labels to binary
            label_mapping = {}
            unique_labels = y.unique()
            
            for label in unique_labels:
                label_lower = str(label).lower()
                if any(term in label_lower for term in ['phish', 'bad', 'malicious', '1', 'fraud']):
                    label_mapping[label] = 1
                else:
                    label_mapping[label] = 0
            
            y = y.map(label_mapping)
            print(f"✅ Label mapping: {label_mapping}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = list(X.columns)
        print(f"✅ Features: {len(self.feature_names)} features")
        
        # Check class distribution
        print(f"\n📊 Class Distribution:")
        print(f"   Legitimate URLs: {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        print(f"   Phishing URLs: {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")
        
        return X, y
    
    def train_ensemble_model(self, X, y):
        """Train an ensemble of models for better accuracy"""
        print("\n" + "=" * 70)
        print("🎯 TRAINING ADVANCED ENSEMBLE MODEL")
        print("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Training set: {len(X_train):,} samples")
        print(f"✅ Test set: {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define multiple models
        print("\n🔧 Initializing models...")
        
        # Model 1: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Model 2: Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        
        # Model 3: XGBoost
        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Create Voting Classifier (Ensemble)
        print("🤖 Creating ensemble with Voting Classifier...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('xgb', xgb_model)
            ],
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        # Train ensemble
        print("🎓 Training ensemble model...")
        ensemble_model.fit(X_train_scaled, y_train)
        print("✅ Training completed!")
        
        # Store the model
        self.model = ensemble_model
        
        # Evaluate
        print("\n" + "=" * 70)
        print("📈 MODEL EVALUATION")
        print("=" * 70)
        
        # Training accuracy
        y_train_pred = ensemble_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"🎯 Training Accuracy: {train_accuracy * 100:.2f}%")
        
        # Test predictions
        y_pred = ensemble_model.predict(X_test_scaled)
        y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'train_accuracy': train_accuracy
        }
        
        print(f"\n📊 Test Set Performance:")
        print(f"   ✅ Accuracy:  {accuracy * 100:.2f}%")
        print(f"   ✅ Precision: {precision * 100:.2f}%")
        print(f"   ✅ Recall:    {recall * 100:.2f}%")
        print(f"   ✅ F1-Score:  {f1 * 100:.2f}%")
        print(f"   ✅ ROC-AUC:   {roc_auc * 100:.2f}%")
        
        # Classification report
        print("\n" + "=" * 70)
        print("📋 DETAILED CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0, 0]:,}")
        print(f"   False Positives: {cm[0, 1]:,}")
        print(f"   False Negatives: {cm[1, 0]:,}")
        print(f"   True Positives:  {cm[1, 1]:,}")
        
        # Feature importance (from Random Forest component)
        print("\n🎯 Calculating feature importance...")
        rf_component = ensemble_model.named_estimators_['rf']
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_component.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 15 Important Features:")
        for idx, row in self.feature_importance.head(15).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def save_model(self):
        """Save the trained model and scaler"""
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL")
        print("=" * 70)
        
        # Save ensemble model
        with open('phishing_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("✅ Model saved: phishing_model.pkl")
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Scaler saved: scaler.pkl")
        
        # Save feature names
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print("✅ Feature names saved: feature_names.pkl")
        
        # Save metrics
        with open('model_metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)
        print("✅ Metrics saved: model_metrics.pkl")
        
        # Save feature importance
        self.feature_importance.to_csv('feature_importance.csv', index=False)
        print("✅ Feature importance saved: feature_importance.csv")
    
    def create_visualizations(self, y_test, y_pred, y_pred_proba):
        """Create and save visualizations"""
        print("\n" + "=" * 70)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 70)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {self.metrics["roc_auc"]:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Feature Importance (Top 15)
        top_features = self.feature_importance.head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='steelblue')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=9)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Metrics Comparison
        metrics_data = {
            'Accuracy': self.metrics['accuracy'],
            'Precision': self.metrics['precision'],
            'Recall': self.metrics['recall'],
            'F1-Score': self.metrics['f1_score'],
            'ROC-AUC': self.metrics['roc_auc']
        }
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        axes[1, 1].bar(metrics_data.keys(), metrics_data.values(), color=colors)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (key, value) in enumerate(metrics_data.items()):
            axes[1, 1].text(i, value + 0.02, f'{value:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: model_performance.png")
        plt.close()
    
    def train(self):
        """Main training pipeline"""
        start_time = datetime.now()
        
        # Load data
        X, y = self.load_and_prepare_data()
        
        # Train model
        X_test, y_test, y_pred, y_pred_proba = self.train_ensemble_model(X, y)
        
        # Create visualizations
        self.create_visualizations(y_test, y_pred, y_pred_proba)
        
        # Save model
        self.save_model()
        
        # Training summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration:.2f} seconds")
        print(f"🎯 Final Accuracy: {self.metrics['accuracy'] * 100:.2f}%")
        print(f"🎯 Final Precision: {self.metrics['precision'] * 100:.2f}%")
        print(f"🎯 Final Recall: {self.metrics['recall'] * 100:.2f}%")
        print(f"✅ Model ready for deployment!")
        print("=" * 70)

if __name__ == '__main__':
    # Train the advanced model
    detector = AdvancedPhishingDetector(
        data_path='PhiUSIIL_Phishing_URL_Dataset.csv',
        min_samples=30000  # Use at least 30k rows
    )
    
    detector.train()
    
    print("\n💡 Next steps:")
    print("   1. Run: streamlit run streamlit_app.py")
    print("   2. Test your phishing detection model!")
