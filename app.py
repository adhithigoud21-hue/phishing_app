from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
from url_features import extract_features
import re

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'phishing_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found. Please train the model first.")
    model = None

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        print(f"\n🔍 Analyzing URL: {url}")
        
        # Extract features and get validation errors
        features, validation_errors = extract_features(url)
        
        print(f"📊 Validation Errors: {validation_errors}")
        
        # If there are critical validation errors, automatically classify as phishing
        if validation_errors:
            result = {
                'prediction': 'phishing',
                'probability': 0.95,  # High confidence it's phishing/invalid
                'features': features,
                'validation_errors': validation_errors,
                'reason': 'URL validation failed - not a real/valid URL'
            }
            print(f"❌ Result: PHISHING (Validation Failed)")
            return jsonify(result)
        
        # Make prediction with model
        feature_values = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(feature_values)[0]
        probability = model.predict_proba(feature_values)[0]
        confidence = probability[prediction]
        
        result = {
            'prediction': 'phishing' if prediction == 1 else 'legitimate',
            'probability': float(confidence),
            'features': features,
            'validation_errors': None
        }
        
        print(f"✅ Result: {result['prediction'].upper()} (confidence: {confidence:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '3.0 - Real URL Validation'
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 ADVANCED PHISHING DETECTION API v3.0")
    print("=" * 70)
    print(f"🌐 Server: http://localhost:5000")
    print(f"📊 Model: Gradient Boosting + DNS Validation")
    print(f"🎯 Features: 30+ detection features + Real URL checking")
    print("=" * 70)
    app.run(debug=True, port=5000, host='0.0.0.0')