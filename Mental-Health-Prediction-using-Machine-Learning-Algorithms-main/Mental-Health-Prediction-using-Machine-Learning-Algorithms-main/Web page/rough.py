import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Create some sample data to test the model
sample_data = np.array([[27, 1, 1]])

# Make predictions on the sample data
prediction = model.predict_proba(sample_data)

# Print the prediction
print(prediction)
