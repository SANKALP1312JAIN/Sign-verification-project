from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import cv2
import base64
from io import BytesIO
from signature_utils import verify_new_signature, extract_signature_from_pdf, detect_signature_regions
import fitz
import sys
sys.path.append('.')  # Ensure current directory is in path for imports
from model_loader import load_model, preprocess_signature

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

# Global variables for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None  # You'll need to load your trained model here

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_new_signature(image_path, target_size=105):
    """
    Preprocess a new signature image for your trained model
    """
    # Use the utility from model_loader for consistency
    return preprocess_signature(image_path, target_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify')
def verify():
    return render_template('verify.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'test')  # 'genuine' or 'test'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename or "uploaded_file")
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Process file based on type
        if file.filename and file.filename.lower().endswith('.pdf'):
            # Handle PDF
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'type': 'pdf',
                'message': 'PDF uploaded successfully. Please select signature area.'
            })
        else:
            # Handle image
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'type': 'image',
                'message': 'Image uploaded successfully.'
            })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/verify', methods=['POST'])
def verify_signatures():
    data = request.get_json()
    
    genuine_file = data.get('genuine_file')
    test_file = data.get('test_file')
    bounding_box = data.get('bounding_box')
    
    if not genuine_file or not test_file:
        return jsonify({'error': 'Both genuine and test signatures are required'}), 400
    
    genuine_path = os.path.join(app.config['UPLOAD_FOLDER'], genuine_file)
    test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_file)
    
    if not os.path.exists(genuine_path) or not os.path.exists(test_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Handle PDF extraction if needed
        if test_file.lower().endswith('.pdf') and bounding_box:
            extracted_img = extract_signature_from_pdf(test_path, bounding_box)
            if extracted_img:
                # Save extracted image
                extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_{test_file}.png")
                extracted_img.save(extracted_path)
                test_path = extracted_path
        
        # Verify signatures using your model
        result, error = verify_new_signature(model, genuine_path, test_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract-pdf', methods=['POST'])
def extract_pdf_signature():
    data = request.get_json()
    pdf_file = data.get('pdf_file')
    bounding_box = data.get('bounding_box')
    
    if not pdf_file:
        return jsonify({'error': 'PDF file is required'}), 400
    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
    
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        extracted_img = extract_signature_from_pdf(pdf_path, bounding_box)
        
        if extracted_img:
            # Convert to base64 for frontend
            buffered = BytesIO()
            extracted_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'image': img_str
            })
        else:
            return jsonify({'error': 'Failed to extract signature'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-signature', methods=['GET', 'POST'])
def detect_signature():
    image_data = None
    candidates = None
    # List all uploaded PDFs for the dropdown
    uploaded_pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    if request.method == 'POST':
        file = request.files.get('document')
        if file and file.filename:
            filename = file.filename.lower()
            # Handle PDF
            if filename.endswith('.pdf'):
                pdf_bytes = file.read()
                pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = pdf[0]
                if hasattr(page, 'get_pixmap'):
                    pix = page.get_pixmap()  # type: ignore[attr-defined]
                    img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
                else:
                    raise AttributeError("PyMuPDF Page object does not have get_pixmap method.")
                image_np = np.array(img)
            else:
                # Image file
                img = Image.open(file.stream).convert('RGB')
                image_np = np.array(img)
            # Detect signature regions
            candidates = detect_signature_regions(image_np)
            # Draw bounding boxes
            vis_img = image_np.copy()
            for cand in candidates:
                x, y, w, h = cand['bbox']
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0,255,0), 2)
            # Convert to PNG and base64 for display
            _, buf = cv2.imencode('.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            image_data = base64.b64encode(buf.tobytes()).decode('utf-8')
    return render_template('detect_signature.html', image_data=image_data, candidates=candidates, uploaded_pdfs=uploaded_pdfs)

@app.route('/detect-signature-auto', methods=['POST'])
def detect_signature_auto():
    image_data = None
    candidates = None
    uploaded_pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    pdf_file = request.form.get('pdf_file')
    if pdf_file:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
        if os.path.exists(pdf_path):
            pdf = fitz.open(pdf_path)
            page = pdf[0]
            if hasattr(page, 'get_pixmap'):
                pix = page.get_pixmap()  # type: ignore[attr-defined]
                img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
            else:
                raise AttributeError("PyMuPDF Page object does not have get_pixmap method.")
            image_np = np.array(img)
            candidates = detect_signature_regions(image_np)
            vis_img = image_np.copy()
            for cand in candidates:
                x, y, w, h = cand['bbox']
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0,255,0), 2)
            _, buf = cv2.imencode('.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            image_data = base64.b64encode(buf.tobytes()).decode('utf-8')
    return render_template('detect_signature.html', image_data=image_data, candidates=candidates, uploaded_pdfs=uploaded_pdfs)

if __name__ == '__main__':
    # Load your model here
    model = load_model('best_signature_model.pth')
    app.run(debug=True, host='0.0.0.0', port=5000) 