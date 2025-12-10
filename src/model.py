import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

class SalesModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.num_feature_indices = []
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Identify numerical features for scaling
        num_cols = ['year', 'month', 'value_sales', 'average_price',
                   'size_numeric', 'price_per_liter', 'brand_frequency']
        self.num_feature_indices = [i for i, col in enumerate(X.columns) if col in num_cols]
        self.feature_names = list(X.columns)
        
        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled.iloc[:, self.num_feature_indices] = self.scaler.fit_transform(
            X_train.iloc[:, self.num_feature_indices]
        )
        X_test_scaled.iloc[:, self.num_feature_indices] = self.scaler.transform(
            X_test.iloc[:, self.num_feature_indices]
        )
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        print(f"Model Training Complete")
        print(f"Test R²: {metrics['r2']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.2f}")
        print(f"Test MAE: {metrics['mae']:.2f}")
        
        return metrics
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = X.copy()
        X_scaled.iloc[:, self.num_feature_indices] = self.scaler.transform(
            X.iloc[:, self.num_feature_indices]
        )
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save(self, model_path, scaler_path):
        """Save model and scaler"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'num_feature_indices': self.num_feature_indices
            }, f)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path, scaler_path):
        """Load model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.num_feature_indices = data['num_feature_indices']
        print(f"Model loaded from {model_path}")
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Calculate model performance metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }