import joblib
import glob
from utils import load_text, prep
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the text data and prepare labels
txt = load_text('test')
texts, labels = [], []

# Process and structure text and labels
for century in sorted(txt):
    for data in txt[century]:
        text = prep(data)  # Process text data
        texts.append(text)
        labels.append(century)

# Encode the labels (centuries)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Load all model files ending in `.joblib`
model_files = glob.glob("*.joblib")
models = {model_file: joblib.load(model_file) for model_file in model_files}

# Sort the model files for consistent output
sorted_model_files = sorted(models.keys())

# Predict the century with each model, print the results, and calculate accuracy
print("Predictions and Accuracy:")
for model_name in sorted_model_files:
    model = models[model_name]
    predictions = []

    # Make predictions for each text
    for text in texts:
        predicted_label = model.predict([text])[0]
        predictions.append(predicted_label)

    # Calculate and display accuracy
    accuracy = accuracy_score(encoded_labels, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

    # Display each prediction if desired
    for text, predicted_label in zip(texts, predictions):
        predicted_century = label_encoder.inverse_transform([predicted_label])[0]
        print(f"{model_name} Prediction for given text: {predicted_century}")

    print("\n")  # Separate outputs for each model
