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

    try {
        console.log('Sending request to /api/predict...');
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        console.log('Predictions:', data.predictions);

        if (data.success && data.predictions && data.predictions.length > 0) {
            const topPrediction = data.predictions[0].crop;
            console.log('Top prediction:', topPrediction);
            
            if (topPrediction) {
                displayResults(topPrediction);
            } else {
                throw new Error('Plant name is undefined');
            }
        } else {
            resultsContent.innerHTML = `
                <div class="result-box" style="border-left-color: #dc3545;">
                    <p style="color: #dc3545; font-weight: 600;">Error: ${data.error || 'Could not identify'}</p>
                </div>
            `;
            results.style.display = 'block';
        }
    } catch (error) {
        console.error('Error details:', error);
        resultsContent.innerHTML = `
            <div class="result-box" style="border-left-color: #dc3545;">
                <p style="color: #dc3545; font-weight: 600;">Error: ${error.message}</p>
                <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Check the console for more details (Press F12)</p>
            </div>
        `;
        results.style.display = 'block';
    }

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
});

function displayResults(prediction) {
    const cleanName = prediction
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
    
    resultsContent.innerHTML = `
        <div class="result-box">
            <div class="result-icon">🌿</div>
            <div class="result-crop">${cleanName}</div>
            <p class="result-subtitle">Your image has been identified!</p>
        </div>
        
        <div class="shop-prompt">
            <p class="shop-prompt-title">Interested in ${cleanName} ?</p>
            <p class="shop-prompt-text">Visit our shop to explore and purchase quality products!</p>
            <button class="btn btn-shop" onclick="window.location.href='/shop'">
                Visit Shop 🛒
            </button>
        </div>
    `;
    results.style.display = 'block';
}