from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer
import os
import sys

# Add the directory containing model.py to the Python path
# This ensures Flask can find your EmotionRNN class
sys.path.append(os.path.dirname(__file__))
from model import EmotionRNN # Import your EmotionRNN class

app = Flask(__name__, static_folder='static', static_url_path='/static')

# --- Configuration for your model ---
# These must EXACTLY match the values used during training!
# If you used 'bert-base-uncased', this vocab size is typical.
VOCAB_SIZE = 30522
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 6 # Number of emotion classes (sadness, joy, love, anger, fear, surprise)
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_LEN = 128 # Max sequence length used during tokenization in training

# Path to your saved model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_rnn_model.pth')

# Labels for interpretation
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model_and_tokenizer():
    """Loads the tokenizer and model when the Flask app starts."""
    global model, tokenizer
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print("Instantiating model...")
        model = EmotionRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)

        print(f"Loading model state dictionary from {MODEL_PATH}...")
        # Load to CPU, as most Flask servers won't have a GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode (important for dropout, batchnorm etc.)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Indicate model loading failure
        tokenizer = None # Indicate tokenizer loading failure
        sys.exit(1) # Exit if model cannot be loaded, as the app won't function

# Call the loading function once when the app starts
with app.app_context():
    load_model_and_tokenizer()

@app.route('/')
def index():
    """Renders the main HTML page for emotion classification."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the UI."""
    print("Received data:", data)  # ✅ Add this
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded. Server might be misconfigured."}), 500

    data = request.get_json(force=True)
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided for prediction"}), 400

    try:
        # Preprocess the input text using the loaded tokenizer
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=False, # Attention mask might not be strictly needed for simple RNN if not used in forward
            return_tensors='pt', # Return PyTorch tensors
        )

        # Move input tensors to the same device as the model (CPU)
        input_ids = encoding['input_ids'].to(torch.device('cpu'))

        # Perform inference
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(input_ids)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_label_id = torch.argmax(probabilities).item()
            predicted_emotion = LABEL_NAMES[predicted_label_id]
            confidence = probabilities[predicted_label_id].item()

        return jsonify({
            "text": text,
            "predicted_emotion": predicted_emotion,
            "confidence": f"{confidence:.4f}",
            "all_probabilities": {name: float(prob) for name, prob in zip(LABEL_NAMES, probabilities)}
        })

    except Exception as e:
        print(f"Prediction failed for text '{text}': {e}")
        return jsonify({"error": f"Prediction failed due to an internal error. Please try again. Details: {e}"}), 500

app = Flask(__name__)
if __name__ == '__main__':
    # Run the Flask app
    # For local development, debug=True is useful.
    # For production, use a WSGI server like Gunicorn or Waitress.
    app.run(host='0.0.0.0', port=5000, debug=True)
