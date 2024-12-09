import numpy as np
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm


def EEGNet(nb_classes, Chans = 34, Samples = 600, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):

    # Select the dropout type (either SpatialDropout2D or Dropout)
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    # Input layer with the shape of EEG data (Channels x Samples)
    input1   = Input(shape = (Chans, Samples, 1))

    # Block 1: Convolution, Batch Normalization, Depthwise Convolution, Activation, Pooling, Dropout
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 10))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    # Block 2: Separable Convolution, Batch Normalization, Activation, Pooling, Dropout
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 10))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    # Flatten the output from the convolutional layers
    flatten      = Flatten(name = 'flatten')(block2)
    
    # Dense layer and softmax activation for classification
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data4EEGNet(kernels, chans, samples):
    # Parameters: Number of kernels, number of channels, number of samples per channel
    # Replace with the correct MAT file path
    mat_file_path = 'D:\\CIRP2025\\experiment\\code\\train\\train.mat'
    # Load the MAT file using scipy.io.loadmat
    loaded_data = scipy.io.loadmat(mat_file_path)

    # Retrieve the data
    data = loaded_data['variables']
    # Print the shape of the data (for debugging)
    print(data.shape)
    
    # Path to the Excel file containing the event labels
    excel_file_path = 'D:\\CIRP2025\\experiment\\code\\train\\event.xlsx'
    df = pd.read_excel(excel_file_path, header=None)

    # Extract labels from the first row of the Excel sheet
    labels = df.iloc[0].values
    # X: EEG data, y: labels
    X = data  #(210,34,600)
    y = labels #(210,3)

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Validation set is just the test set in this case
    X_validate   = X_test
    Y_validate   = Y_test
    
    # Convert labels to one-hot encoding
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)

    # Reshape the input data to match the input shape expected by the model
    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

import tensorflow as tf
import time
# Start the timer
start_time = time.time()

kernels, chans, samples = 1, 34, 600

X_train, X_validate, X_test, Y_train, Y_validate, Y_test = get_data4EEGNet(kernels, chans, samples)

# Initialize the EEGNet model
model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# Compile the model with categorical crossentropy loss and adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# Define a variable to track the best accuracy
best_acc = 0

# Define a callback function to save the best model
def save_best_model(epoch, logs):
    global best_acc
    # Get the current validation accuracy
    acc = logs['val_accuracy']
    # If the current accuracy is better than the previous best, save the model
    if acc > best_acc:
        best_acc = acc
        model.save('D:\\CIRP2025\\experiment\\code\\DataModel\\bestmodel.h5')
        print(f'\nEpoch {epoch+1}: best model saved with val_accuracy: {best_acc:.4f}')

# Create a custom callback object for saving the best model
save_best_model_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_best_model)

# Class weights (used to handle class imbalance, if needed)
class_weights = {0:1, 1:1, 2:1, 3:1}

# Train the model with the training data, using the validation set for monitoring performance
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 200, 
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[save_best_model_callback], class_weight = class_weights)

# Load the best saved model after training
model = tf.keras.models.load_model('D:\\CIRP2025\\experiment\\code\\DataModel\\bestmodel.h5')

# Predict on the test set
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_true=Y_test.argmax(axis=-1), y_pred=preds)

# Normalize the confusion matrix to percentages
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot the normalized confusion matrix as a heatmap
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('D:\\CIRP2025\\experiment\\code\\DataModel\\confusion_matrix.png')  # Save the figure
plt.show()

# End the timer
end_time = time.time()
# Calculate and print the total runtime
run_time = end_time - start_time
print("Code execution time: ", run_time, "seconds")