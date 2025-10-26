# 🌱 SeedLink - Agricultural Crop Classifier

AI-powered crop classification system using deep learning.

## Features
- 🌾 Classifies 96 different agricultural crops
- 🎯 85%+ accuracy using EfficientNetV2-M
- 🖼️ Drag & drop image upload interface
- 🛒 E-commerce recommendation system
- 🔌 REST API for MERN stack integration

## Tech Stack
- **Backend:** Flask, PyTorch
- **Frontend:** HTML, CSS, JavaScript
- **Model:** EfficientNetV2-M (Transfer Learning)
- **Dataset:** 5,000+ crop images

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Setup
\`\`\`bash
# Clone repository
git clone https://github.com/yourusername/seedlink.git
cd seedlink

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

## Usage

### Run Flask App
\`\`\`bash
python app.py
\`\`\`

Visit: http://localhost:5000

### API Endpoints
- `POST /api/predict` - Predict crop from image
- `GET /api/health` - Health check
- `GET /api/classes` - Get all crop classes
- `GET /api/recommendations/<crop>` - Get product recommendations

## Model Performance
- Training Accuracy: 97.2%
- Validation Accuracy: 95.8%
- Test Accuracy: 94.3%

## Project Structure
\`\`\`
seedlink/
├── app.py                 # Flask application
├── templates/            # HTML templates
├── static/              # CSS, JS, images
├── requirements.txt     # Python dependencies
├── best_model.pth      # Trained model (download separately)
└── classes.pkl         # Class names
\`\`\`

## Model Download
Due to file size limitations, download the trained model from:
- [Google Drive Link] or [Release Page]

## License
MIT License

## Contributors
- Your Name (@yourgithub)

## Acknowledgments
- PyTorch Team
- Agricultural dataset providers
\`\`\`

---

## 🎨 Step 10: Add README and Push
```bash
# Add README
git add README.md

# Commit
git commit -m "docs: Add comprehensive README"

# Push
git push
```

---

## 🔍 Step 11: Verify Upload

1. Go to your GitHub repository: `https://github.com/RheaRoche/SeedLink-airecognition.git`
2. Check that files are there
3. Verify `.gitignore` is working (large files should NOT be uploaded)

---

## ⚠️ Common Issues & Solutions

### Issue 1: Large files error
