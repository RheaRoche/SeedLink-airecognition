# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from PIL import Image
import pickle
import os
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = r'D:\SeedLink\best_model.pth'
CLASSES_PATH = r'D:\SeedLink\classes.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load classes
print("Loading classes...")
with open(CLASSES_PATH, 'rb') as f:
    classes = pickle.load(f)
print(f"✓ {len(classes)} classes loaded")

# Load checkpoint
print("Loading model checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = len(classes)

# Build EfficientNetV2-M architecture
print("Building model architecture...")
model = efficientnet_v2_m(weights=None)

# Rebuild classifier to match training architecture
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Model loaded successfully!")
print(f"✓ Best validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_file, threshold=0.30):
    """Predict crop from image file with confidence threshold"""
    try:
        # Load and preprocess image
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probs, min(5, num_classes))
            
            predictions = []
            for i in range(top5_probs.size(1)):
                crop_name = classes[top5_indices[0][i].item()]
                confidence = float(top5_probs[0][i].item() * 100)
                predictions.append({
                    'crop': crop_name,
                    'confidence': confidence
                })
            
            # Check if top prediction meets threshold
            is_confident = predictions[0]['confidence'] >= (threshold * 100)
            
            return {
                'success': True,
                'is_confident': is_confident,
                'predictions': predictions,
                'threshold': threshold * 100,
                'top_prediction': predictions[0]['crop'],
                'top_confidence': predictions[0]['confidence']
            }
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    print("\n=== Received prediction request ===")
    
    if 'image' not in request.files:
        print("Error: No image in request")
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    print(f"File received: {file.filename}")
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    # Get confidence threshold from request (default 30%)
    threshold = float(request.form.get('threshold', 0.30))
    
    # Predict
    print("Running prediction...")
    result = predict_image(file, threshold)
    
    if result['success']:
        print(f"✓ Prediction successful: {result['top_prediction']} ({result['top_confidence']:.2f}%)")
    else:
        print(f"✗ Prediction failed: {result.get('error', 'Unknown error')}")
    
    return jsonify(result)

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available crop classes"""
    return jsonify({
        'success': True,
        'classes': classes,
        'total': len(classes)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'running',
        'model': 'EfficientNetV2-M',
        'model_loaded': True,
        'device': str(device),
        'classes': len(classes),
        'val_accuracy': f"{checkpoint.get('val_acc', 0):.2f}%"
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌱 SEEDLINK - AI CROP CLASSIFIER")
    print("="*60)
    print(f"Model: EfficientNetV2-M")
    print(f"Classes: {len(classes)}")
    print(f"Device: {device}")
    print(f"Validation Accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    print("="*60)
    print("\n🚀 Starting Flask server...")
    print("📱 Open your browser and navigate to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run on all interfaces to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=True)