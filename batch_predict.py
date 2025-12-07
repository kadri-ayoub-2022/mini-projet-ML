"""
Batch prediction script for multiple countries
Upload a CSV file and get predictions for all rows
"""

import pandas as pd
import pickle
import sys

def load_model(model_path='best_model.pkl'):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_batch(input_csv, output_csv='predictions.csv'):
    """
    Make predictions for all rows in input CSV

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save predictions

    Expected CSV columns:
        happiness, income, sanitation, water, literacy, inequality,
        energy, childmortality, fertility, hiv, foodsupply, population
    """

    print("\n" + "="*70)
    print("BATCH PREDICTION - Life Expectancy Predictor")
    print("="*70)

    # Load model
    print("\n[1/4] Loading model...")
    model_package = load_model()
    print(f"   ✓ Model loaded: {model_package['model_name']}")
    print(f"   ✓ Performance: MAE={model_package['performance']['mae']:.2f}, R²={model_package['performance']['r2']:.4f}")

    # Load input data
    print(f"\n[2/4] Loading input data from '{input_csv}'...")
    try:
        data = pd.read_csv(input_csv)
        print(f"   ✓ Loaded {len(data)} rows")
    except FileNotFoundError:
        print(f"   ✗ Error: File '{input_csv}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        sys.exit(1)

    # Verify columns
    required_features = model_package['feature_names']
    missing_cols = [col for col in required_features if col not in data.columns]

    if missing_cols:
        print(f"   ✗ Error: Missing required columns: {missing_cols}")
        print(f"   Required columns: {required_features}")
        sys.exit(1)

    print(f"   ✓ All required features present")

    # Make predictions
    print("\n[3/4] Making predictions...")

    # Extract features in correct order
    X = data[required_features]

    # Scale features
    X_scaled = model_package['scaler'].transform(X)

    # Predict
    predictions = model_package['model'].predict(X_scaled)

    # Add predictions to dataframe
    result = data.copy()
    result['predicted_lifeexp'] = predictions
    result['confidence_lower'] = predictions - model_package['performance']['mae']
    result['confidence_upper'] = predictions + model_package['performance']['mae']

    print(f"   ✓ Predictions completed for {len(predictions)} rows")

    # Save results
    print(f"\n[4/4] Saving results to '{output_csv}'...")
    result.to_csv(output_csv, index=False)
    print(f"   ✓ Results saved successfully")

    # Display summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"\nPredicted Life Expectancy Statistics:")
    print(f"   Minimum:  {predictions.min():.2f} years")
    print(f"   Maximum:  {predictions.max():.2f} years")
    print(f"   Average:  {predictions.mean():.2f} years")
    print(f"   Std Dev:  {predictions.std():.2f} years")

    print(f"\nResults saved to: {output_csv}")
    print("="*70 + "\n")

    return result

def create_sample_csv(filename='sample_batch_input.csv'):
    """Create a sample CSV file for testing"""
    sample_data = pd.DataFrame({
        'country_name': ['Country A', 'Country B', 'Country C'],
        'happiness': [7.5, 5.0, 3.5],
        'income': [45000, 15000, 2000],
        'sanitation': [99.0, 75.0, 30.0],
        'water': [100.0, 85.0, 50.0],
        'literacy': [99.0, 80.0, 40.0],
        'inequality': [30.0, 40.0, 55.0],
        'energy': [5000, 1500, 300],
        'childmortality': [5.0, 30.0, 80.0],
        'fertility': [1.8, 3.0, 5.5],
        'hiv': [20000, 100000, 500000],
        'foodsupply': [3300, 2500, 1800],
        'population': [10000000, 30000000, 15000000]
    })

    sample_data.to_csv(filename, index=False)
    print(f"✓ Sample CSV created: {filename}")
    return filename

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch Life Expectancy Prediction')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default='predictions.csv',
                       help='Output CSV file path (default: predictions.csv)')
    parser.add_argument('--sample', action='store_true',
                       help='Create a sample input CSV file')

    args = parser.parse_args()

    if args.sample:
        # Create sample CSV
        sample_file = create_sample_csv()
        print(f"\nYou can now run: python batch_predict.py -i {sample_file}")
    elif args.input:
        # Run batch prediction
        predict_batch(args.input, args.output)
    else:
        # Show help
        print("\nBatch Life Expectancy Prediction Tool")
        print("="*50)
        print("\nUsage:")
        print("  1. Create sample CSV:")
        print("     python batch_predict.py --sample")
        print("\n  2. Run predictions:")
        print("     python batch_predict.py -i input.csv -o output.csv")
        print("\n  3. Show this help:")
        print("     python batch_predict.py")
        print("\nRequired CSV columns:")
        print("  happiness, income, sanitation, water, literacy,")
        print("  inequality, energy, childmortality, fertility,")
        print("  hiv, foodsupply, population")
        print("="*50 + "\n")
