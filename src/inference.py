import pandas as pd
from .data_processing import DataProcessor
from .feature_engineering import FeatureEngineer
from .model import SalesModel

class PredictionPipeline:
    def __init__(self, model_path, scaler_path, encoders_path):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model = SalesModel()
        
        # Load model
        self.model.load(model_path, scaler_path)
        
        # Load encoders
        import pickle
        with open(encoders_path, 'rb') as f:
            data = pickle.load(f)
            self.feature_engineer.encoders = data['encoders']
            self.feature_engineer.brand_freq_map = data['brand_freq_map']
    
    def predict_single(self, record):
        """Predict for a single record"""
        # Validate input
        df = self.data_processor.validate_single_record(record)
        
        # Clean data
        df = self.data_processor.clean_data(df, is_training=False)
        
        # Engineer features
        df = self.feature_engineer.create_features(df, is_training=False)
        df = self.feature_engineer.encode_features(df, is_training=False)
        
        # Prepare features
        feature_names = self.feature_engineer.get_feature_names()
        X = df[feature_names]
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        return {
            'predicted_volume_sales': float(prediction),
            'input_data': record if isinstance(record, dict) else record.to_dict('records')[0]
        }
