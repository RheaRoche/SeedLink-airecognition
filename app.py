# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
import os
import base64
from io import BytesIO
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # Enable CORS for MERN integration

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'best_model.pth'
CLASSES_PATH = 'classes.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classes
with open(CLASSES_PATH, 'rb') as f:
    classes = pickle.load(f)

# Build model architecture
model = models.resnet50(pretrained=False)
num_classes = len(classes)

# Rebuild the same classifier structure
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print(f"✓ Model loaded successfully on {device}")
print(f"✓ {num_classes} classes loaded")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_file, threshold=0.75):
    """Predict crop from image file"""
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
                predictions.append({
                    'crop': classes[top5_indices[0][i].item()],
                    'confidence': float(top5_probs[0][i].item() * 100)
                })
            
            # Check if top prediction meets threshold
            is_confident = predictions[0]['confidence'] >= (threshold * 100)
            
            return {
                'success': True,
                'is_confident': is_confident,
                'predictions': predictions,
                'threshold': threshold * 100
            }
    
    except Exception as e:
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
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    # Get confidence threshold from request
    threshold = float(request.form.get('threshold', 0.75))
    
    # Predict
    result = predict_image(file, threshold)
    
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
        'model': 'loaded',
        'device': str(device),
        'classes': len(classes)
    })

if __name__ == '__main__':
    # Run on all interfaces to allow external connections
    app.run(host='0.0.0.0', port=5000)