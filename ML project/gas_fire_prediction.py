import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Load the dataset
file_path = 'gas_sensor.csv'  # Update with your file path
data = pd.read_excel(file_path)

# Encode the 'Gas' column to numerical values
label_encoder = LabelEncoder()
data['Gas'] = label_encoder.fit_transform(data['Gas'])

# Generate additional synthetic data to increase training set
def generate_synthetic_data(data, n_samples=1000):
    # Resample the data to create synthetic samples
    synthetic_data = resample(data, replace=True, n_samples=n_samples, random_state=42)
    
    # Separate numerical and categorical columns
    numeric_columns = ['Gas Value', 'Gas Increase Rate', 'Temperature']
    categorical_columns = ['Gas', 'Fire state']
    
    # Add noise only to the numeric columns
    noise = np.random.normal(0, 0.1, synthetic_data[numeric_columns].shape)
    synthetic_data[numeric_columns] = synthetic_data[numeric_columns] + noise
    
    # Clip values to ensure they remain within reasonable bounds
    synthetic_data['Gas Value'] = np.clip(synthetic_data['Gas Value'], 0, None)
    synthetic_data['Gas Increase Rate'] = np.clip(synthetic_data['Gas Increase Rate'], 0, None)
    synthetic_data['Temperature'] = np.clip(synthetic_data['Temperature'], 0, None)
    
    return synthetic_data

# Append synthetic data to the original dataset
synthetic_data = generate_synthetic_data(data)
data = pd.concat([data, synthetic_data], ignore_index=True)

# Prepare features (X) and target (y)
X = data[['Gas Value', 'Gas Increase Rate', 'Temperature', 'Gas']]
y_fire = data['Fire state'].apply(lambda x: 1 if x.lower() == 'fire' else 0)
y_gas = data['Gas']

# Split the data into training and testing sets for both fire and gas prediction
X_train, X_test, y_fire_train, y_fire_test, y_gas_train, y_gas_test = train_test_split(
    X, y_fire, y_gas, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the neural network for fire prediction
mlp_fire = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp_fire.fit(X_train, y_fire_train)

# Build and train the neural network for gas type prediction
mlp_gas = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp_gas.fit(X_train, y_gas_train)

# Predict on the test set
y_fire_pred = mlp_fire.predict(X_test)
y_gas_pred = mlp_gas.predict(X_test)

# Evaluate the models
fire_accuracy = accuracy_score(y_fire_test, y_fire_pred)
gas_accuracy = accuracy_score(y_gas_test, y_gas_pred)
fire_report = classification_report(y_fire_test, y_fire_pred)
gas_report = classification_report(y_gas_test, y_gas_pred, target_names=label_encoder.classes_)

# Print results
print(f"Fire Prediction Accuracy: {fire_accuracy:.2f}")
print("Fire Classification Report:")
print(fire_report)
print(f"Gas Prediction Accuracy: {gas_accuracy:.2f}")
print("Gas Classification Report:")
print(gas_report)

# Function to predict fire and gas type based on user input
def predict_fire_and_gas(gas_value, gas_increase_rate, temperature):
    # Create input data with a placeholder for the gas type
    placeholder_gas_type = 0  # Assuming 0 is a valid placeholder
    input_data = np.array([[gas_value, gas_increase_rate, temperature, placeholder_gas_type]])
    input_data = scaler.transform(input_data)
    
    fire_prediction = mlp_fire.predict(input_data)[0]
    gas_prediction = mlp_gas.predict(input_data)[0]
    
    fire_probability = mlp_fire.predict_proba(input_data)[0][1]
    gas_type = label_encoder.inverse_transform([gas_prediction])[0]
    
    return fire_probability, gas_type

# Example usage
gas_value = 400
gas_increase_rate = 50
temperature = 42

fire_probability, gas_type = predict_fire_and_gas(gas_value, gas_increase_rate, temperature)
print(f"Fire Probability: {fire_probability:.2f}")
print(f"Predicted Gas Type: {gas_type}")
