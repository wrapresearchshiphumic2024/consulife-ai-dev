from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

MODEL_PATH = "./saved_model/best_model"
nltk_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in nltk_stopwords)
    return text

# Load the tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text input is required."}), 400

        preprocessed_text = preprocess_text(text)

        # Tokenize and encode the input text
        inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=64)

        # Predict probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.sigmoid(logits).squeeze().tolist()

        class_names = ['Stress', 'Anxiety', 'Depression']

        response = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
