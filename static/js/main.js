// main.js
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const results = document.getElementById('results');
const resultsContent = document.getElementById('resultsContent');

let selectedFile = null;

// Click to upload
dropzone.addEventListener('click', () => fileInput.click());

// File selection
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('Please select a valid image file (PNG, JPG, or JPEG)');
    }
});

// Drag and drop events
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('Please drop a valid image file (PNG, JPG, or JPEG)');
    }
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropzone.style.display = 'none';
        preview.style.display = 'block';
        analyzeBtn.disabled = false;
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    dropzone.style.display = 'block';
    preview.style.display = 'none';
    analyzeBtn.disabled = true;
    results.style.display = 'none';
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('threshold', '0.30'); // 30% confidence threshold

    try {
        console.log('Sending request to /api/predict...');
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);

        if (data.success) {
            if (data.predictions && data.predictions.length > 0) {
                displayResults(data);
            } else {
                throw new Error('No predictions returned');
            }
        } else {
            displayError(data.error || 'Could not identify the plant');
        }
    } catch (error) {
        console.error('Error details:', error);
        displayError(`Failed to analyze image: ${error.message}`);
    }

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
});

function displayResults(data) {
    const topPrediction = data.predictions[0];
    const cleanName = topPrediction.crop
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    
    let resultsHTML = `
        <div class="result-box">
            <div class="result-icon">🌿</div>
            <div class="result-crop">${cleanName}</div>
            <p class="result-subtitle">Your image has been identified!</p>
        </div>
        
        <div class="shop-prompt">
            <p class="shop-prompt-title">Interested in ${cleanName}?</p>
            <p class="shop-prompt-text">Visit our shop to explore and purchase quality products!</p>
            <button class="btn btn-shop" onclick="window.location.href='/shop'">
                Visit Shop 🛒
            </button>
        </div>
    `;
    
    resultsContent.innerHTML = resultsHTML;
    results.style.display = 'block';
    
    // Smooth scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(errorMessage) {
    resultsContent.innerHTML = `
        <div class="result-box error-box">
            <div class="error-icon">❌</div>
            <p class="error-title">Analysis Failed</p>
            <p class="error-message">${errorMessage}</p>
            <p class="error-suggestion">
                Please try again with:
                <br>• A clearer image
                <br>• Better lighting
                <br>• A closer view of the plant
            </p>
        </div>
    `;
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check server health on page load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Server health:', data);
        
        if (data.success) {
            console.log(`✓ Model loaded: ${data.model}`);
            console.log(`✓ Classes: ${data.classes}`);
            console.log(`✓ Validation Accuracy: ${data.val_accuracy}`);
        }
    } catch (error) {
        console.error('Server health check failed:', error);
    }
});