from keras.models import load_model
import scipy.io
import numpy as np

# Load the trained model
model = load_model('D:\\CIRP2025\\experiment\\code\\DataModel\\bestmodel.h5')

# Load the data from a .mat file
mat_file_path = 'D:\\CIRP2025\\experiment\\code\\test\\car1.mat'
loaded_data = scipy.io.loadmat(mat_file_path)

# Extract the EEG data from the loaded file
data = loaded_data['variables']
# Reshape the data to fit the model input shape (1 sample, 34 channels, 8400 samples)
data = data.reshape(1, 34, 8400)
print("Data shape:", data.shape)

# Define window size and stride for sliding window approach
window_size = 600
stride = 600

# Initialize a list to store all predictions for each recognition process
all_predictions = []

# Iterate over the first dimension (usually 1, as the data is reshaped to (1, 34, 8400))
for i in range(data.shape[0]):
    # Initialize a list to store predictions for all windows in the current recognition process
    process_predictions = []
    
    # Slide the window along the third dimension (time axis)
    for j in range(0, data.shape[2] - window_size + 1, stride):
        # Extract the window of data (one segment of EEG signals)
        window_data = data[i, :, j:j+window_size].reshape(1, 34, 600)
        
        # Make predictions using the model
        predictions = model.predict(window_data)
        
        # Convert the prediction into a label (argmax returns the index of the highest probability)
        predicted_label = np.argmax(predictions) + 1  # Add 1 to match the label range (1-based index)
        process_predictions.append(predicted_label)
    
    # Add the predictions for the current process to the list of all predictions
    all_predictions.append(process_predictions)

# Output the prediction results for each recognition process
for i, predictions in enumerate(all_predictions):
    print(f"All window predictions for recognition process {i+1}:")
    for j, label in enumerate(predictions):
        print(f"Prediction for window {j+1} is label {label}")
