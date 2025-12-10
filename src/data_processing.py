import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        # SKU is not required
        self.required_columns = [
            'city', 'store_name', 'manufacturer', 'brand', 'class', 
            'size', 'price_bracket', 'year', 'month', 'value_sales', 
            'volume_sales', 'average_price'
        ]
    
    def load_data(self, file_path):
        """Load CSV file and validate columns"""
        df = pd.read_csv(file_path)
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df
    
    def clean_data(self, df, is_training=True):
        """Clean and validate data"""
        df = df.copy()
        
        # Remove invalid records (only for training)
        if is_training:
            df = df[df['volume_sales'] > 0]
        
        # Handle missing values (if founded)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def validate_single_record(self, record):
        """Validate a single record for prediction"""
        if isinstance(record, dict):
            record = pd.DataFrame([record])
        
        missing_cols = set(self.required_columns) - set(record.columns)
        if 'volume_sales' in missing_cols:
            missing_cols.remove('volume_sales')
        
        if missing_cols:
            raise ValueError(f"Missing required fields: {missing_cols}")
        
        return record