import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}
        self.categorical_cols = ['city', 'manufacturer', 'brand', 'class', 'season']
        self.numerical_cols = ['year', 'month', 'value_sales', 'average_price',
                              'size_numeric', 'price_per_liter', 'brand_frequency']
        self.brand_freq_map = {}
    
    def create_features(self, df, is_training=True):
        """Generate all engineered features"""
        df = df.copy()
        
        # Extract numeric size
        df['size_numeric'] = df['size'].apply(self._extract_numeric_size)
        
        # Price per liter
        df['price_per_liter'] = df['average_price'] / df['size_numeric']
        
        # Brand frequency
        if is_training:
            self.brand_freq_map = df['brand'].value_counts().to_dict()
        df['brand_frequency'] = df['brand'].map(self.brand_freq_map).fillna(1)
        
        # Season
        df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def encode_features(self, df, is_training=True):
        """Encode categorical variables"""
        df = df.copy()
        
        for col in self.categorical_cols:
            if is_training:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_enc'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df[f'{col}_enc'] = df[col].apply(
                    lambda x: self._safe_transform(col, x)
                )
        
        return df
    
    def get_feature_names(self):
        """Return final feature column names"""
        encoded_cols = [f'{col}_enc' for col in self.categorical_cols]
        return self.numerical_cols + encoded_cols
    
    @staticmethod
    def _extract_numeric_size(size_str):
        """Extract numeric value from size string"""
        try:
            match = re.search(r'(\d+\.?\d*)', str(size_str))
            return float(match.group(1)) if match else 1.0
        except:
            return 1.0
    
    @staticmethod
    def _get_season(month):
        """Map month to season"""
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        return season_map.get(month, 'Unknown')
    
    def _safe_transform(self, col, value):
        """Handle encoding with unknown categories"""
        encoder = self.encoders[col]
        try:
            return encoder.transform([str(value)])[0]
        except ValueError:
            # Return most frequent class for unseen values
            return encoder.transform([encoder.classes_[0]])[0]
