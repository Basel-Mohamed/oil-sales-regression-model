import sys
import pickle
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model import SalesModel

def train_pipeline(data_path, model_dir='models'):
    print("Starting Training Pipeline\n")
    
    # Initialize components
    processor = DataProcessor()
    engineer = FeatureEngineer()
    model = SalesModel()
    
    # Load and process data
    print("Loading data...")
    df = processor.load_data(data_path)
    print(f"Loaded {len(df)} records")
    
    # Clean data
    print("Cleaning data...")
    df = processor.clean_data(df, is_training=True)
    print(f"After cleaning: {len(df)} records")
    
    # Engineer features
    print("Engineering features...")
    df = engineer.create_features(df, is_training=True)
    df = engineer.encode_features(df, is_training=True)
    
    # Prepare features and target
    feature_names = engineer.get_feature_names()
    X = df[feature_names]
    y = df['volume_sales']
    print(f"Feature matrix: {X.shape}")
    
    # Train model
    print("\nTraining model...")
    metrics = model.train(X, y)
    
    # Save artifacts
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(
        f'{model_dir}/random_forest_model.pkl',
        f'{model_dir}/scaler.pkl'
    )
    
    with open(f'{model_dir}/encoders.pkl', 'wb') as f:
        pickle.dump({
            'encoders': engineer.encoders,
            'brand_freq_map': engineer.brand_freq_map
        }, f)
    
    print(f"\nTraining completed. Models saved to {model_dir}/")
    return metrics

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data\oil_sales_assignment_dataset.csv'
    train_pipeline(data_path)