# Plant Disease Prediction System

A deep learning-powered web application that identifies plant diseases from uploaded leaf images. Built with Django and PyTorch, this application leverages a Convolutional Neural Network (CNN) to predict the crop species and its specific disease state, returning real-time confidence metrics.

## Architecture Overview

\\\
User Interface (Web Browser)
    │
    ▼
┌──────────────────────────┐
│  Image Upload via Form   │  ← Receives leaf image
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Django Request Router (predictor/views.py)  │
└──────────┬───────────────────────────────────┘
           │ (Passes Raw Image Bytes)
           ▼
┌──────────────────────────────────────────────┐
│  Data Preprocessing (ml_files/utils.py)      │
│  - PIL Image Load & RGB Conversion           │
│  - Resize (128x128) & ToTensor               │
│  - Normalize (ImageNet Means/Stds)           │
└──────────┬───────────────────────────────────┘
           │ (Tensors)
           ▼
┌──────────────────────────────────────────────┐
│  PyTorch SimpleCNN Model                     │
│  - Evaluates pre-trained plant_disease_model │
│  - Applies Softmax for Class Probabilities   │
└──────────┬───────────────────────────────────┘
           │
           ▼
        Prediction Result
  (e.g., "Apple: Black rot", 95.2% Confidence)
\\\

## System Output

The system processes uploaded images and immediately outputs a structured prediction payload in JSON format, which is rendered dynamically on the front end:

| Uploaded Image | Predicted Class (Plant: Disease) | Confidence |
|---|---|---|
| leaf_sample1.jpg | Apple: Black rot | 98.45% |
| leaf_sample2.png | Tomato: Early blight | 87.12% |

## Directory Structure

\\\
├── manage.py                   # Django CLI entrypoint
├── db.sqlite3                  # Django SQLite database
├── README.md                   # Project documentation
│
├── plant_project/              # Core Django settings and routing
│
└── predictor/                  # Main application module
    ├── views.py                # View controllers for UI and inference endpoints
    ├── ml_files/               # Machine Learning Sub-module
    │   ├── class_names.json    # JSON mapping of model outputs to human-readable labels
    │   ├── model.py            # PyTorch SimpleCNN architecture definition
    │   ├── plant_disease_model.pth # Saved weights for the trained neural network
    │   └── utils.py            # Image transforms and inference execution logic
    │
    ├── templates/              # HTML frontend templates (index.html)
    └── static/                 # Static assets (CSS, JS, images)
\\\

## How to Run

### 1. Prerequisites
- Python 3.9+
- PyTorch (CPU or CUDA-enabled)
- Django

### 2. Install Dependencies

\\\ash
# Create and activate environment (recommended)
python -m venv venv
source venv/bin/activate  # Or env\Scripts\activate on Windows

# Install required packages
pip install django torch torchvision pillow
\\\

### 3. Run the Application

\\\ash
# Apply migrations (if needed)
python manage.py migrate

# Launch the Django web server
python manage.py runserver
\\\

### 4. Usage
1. Open the application in your browser at http://127.0.0.1:8000.
2. Click on the file upload button to select a photo of a plant leaf.
3. Submit the image.
4. The system will asynchronously pass the image to the PyTorch backend, replacing the underscores in the class name with a readable format (e.g., \Apple___Black_rot\ to \Apple: Black rot\) and displaying the confidence score.

## Key Design Decisions

1. **In-Memory Image Processing**: To avoid disk I/O bottlenecks and temporary file clutter, uploaded images are passed as raw bytes \io.BytesIO()\ directly from the Django \equest.FILES\ object into the PIL processing pipeline.
2. **Coupled PyTorch Backend**: The PyTorch model is loaded globally into memory when the Django server starts (via \utils.py\). This prevents the application from enduring the massive overhead of initializing the model for every single user request.
3. **Hardware Agnostic**: The model uses \map_location=device\ to seamlessly run on CUDA GPUs if available, but falls back safely to CPU execution for standard web hosting environments.
