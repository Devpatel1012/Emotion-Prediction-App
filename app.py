from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer
import os
import sys

# Add the directory containing model.py to the Python path
sys.path.append(os.path.dirname(__file__))
from model import EmotionRNN  # Import your EmotionRNN class

app = Flask(__name__, static_folder='static', static_url_path='/static')

# --- Model Configuration ---
VOCAB_SIZE = 30522
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 6
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_LEN = 128

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_rnn_model.pth')
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# --- Load model and tokenizer once globally ---
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    try:
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print("🧠 Instantiating model...")
        model = EmotionRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)

        print(f"📥 Loading model weights from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        model = None
        tokenizer = None
        sys.exit(1)

# Load model once when app starts
with app.app_context():
    load_model_and_tokenizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded."}), 500

    data = request.get_json(force=True)
    print("📨 Received data:", data)

    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(torch.device('cpu'))

        with torch.no_grad():
            outputs = model(input_ids)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_label_id = torch.argmax(probabilities).item()
            predicted_emotion = LABEL_NAMES[predicted_label_id]
            confidence = probabilities[predicted_label_id].item()

        return jsonify({
            "text": text,
            "predicted_emotion": predicted_emotion,
            "confidence": f"{confidence:.4f}",
            "all_probabilities": {
                name: float(prob) for name, prob in zip(LABEL_NAMES, probabilities)
            }
        })

    except Exception as e:
        print(f"❌ Prediction failed for input '{text}': {e}")
        return jsonify({"error": f"Prediction failed. Error: {e}"}), 500

if __name__ == '__main__':
    # Works for local dev, but use gunicorn on Render
    app.run(host='0.0.0.0', port=5000, debug=True)
