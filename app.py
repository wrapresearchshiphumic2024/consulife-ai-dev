from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = "./saved_model/best_model.pkl"
VECTORIZER_PATH = "./saved_model/vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text input is required."}), 400

        preprocessed_text = preprocess_text(text)

        transformed_text = vectorizer.transform([preprocessed_text])

        probabilities = model.predict_proba(transformed_text)[0]

        class_names = ['Stress', 'Anxiety', 'Depression']

        response = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import re
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in nltk_stopwords)
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))
