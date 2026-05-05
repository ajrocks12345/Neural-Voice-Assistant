from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle

# Import necessary classes for unpickling
from lstm_numpy import LSTMRNN
from nlp_preprocessor import NLPPreprocessor
from action_executor import execute_action

app = Flask(__name__)
CORS(app)

# Load the model and preprocessor globally
try:
    with open('intent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        prep = pickle.load(f)
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError:
    print("WARNING: Model files not found. Run train_intent_model.py first.")
    model = None
    prep = None



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_intent():
    if not model or not prep:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    
    # 1. Preprocess
    seq = prep.text_to_sequence(text)
    onehot_seq = prep.sequence_to_onehot(seq)
    
    # 2. Predict with LSTM
    pred_idx = model.predict(onehot_seq)
    intent = prep.idx_to_intent[pred_idx]
    
    # 3. Action Response (using the new executor)
    action_response = execute_action(intent, text)
    
    return jsonify({
        "text": text,
        "intent": intent,
        "action": action_response
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
