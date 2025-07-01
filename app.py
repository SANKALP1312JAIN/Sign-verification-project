from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, make_response
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
import tempfile
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

@app.route('/get-pdf-page-image')
def get_pdf_page_image():
    pdf_file = request.args.get('pdf_file')
    if not pdf_file:
        return jsonify({'error': 'No PDF file specified'}), 400
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'File not found'}), 404
    pdf = fitz.open(pdf_path)
    page = pdf[0]
    pix = page.get_pixmap()  # type: ignore[attr-defined]
    img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return jsonify({'image': f'data:image/png;base64,{img_str}'})

@app.route('/extract-signature-area', methods=['POST'])
def extract_signature_area():
    pdf_file = request.form.get('pdf_file')
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    width = int(request.form.get('width', 0))
    height = int(request.form.get('height', 0))
    uploaded_pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    image_data = None
    candidates = None
    if pdf_file and width > 0 and height > 0:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
        if os.path.exists(pdf_path):
            pdf = fitz.open(pdf_path)
            page = pdf[0]
            pix = page.get_pixmap()  # type: ignore[attr-defined]
            img = Image.open(BytesIO(pix.tobytes("png"))).convert('RGB')
            # Crop the selected area
            cropped = img.crop((x, y, x + width, y + height))
            buffered = BytesIO()
            cropped.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data = img_str
    # Show the cropped image as the result
    return render_template('detect_signature.html', image_data=image_data, candidates=None, uploaded_pdfs=uploaded_pdfs)

@app.route('/verify-cropped-signature', methods=['POST'])
def verify_cropped_signature():
    uploaded_pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
    image_data = None
    candidates = None
    result = None
    error = None
    # Get cropped image (base64)
    cropped_image_data = request.form.get('cropped_image_data')
    real_signature_file = request.files.get('real_signature')
    if cropped_image_data and real_signature_file:
        # Save cropped image to temp file
        cropped_bytes = base64.b64decode(cropped_image_data)
        cropped_path = os.path.join(tempfile.gettempdir(), f"cropped_{uuid.uuid4()}.png")
        with open(cropped_path, 'wb') as f:
            f.write(cropped_bytes)
        # Save real signature to temp file
        real_sig_filename = secure_filename(real_signature_file.filename or "real_signature.png")
        real_sig_path = os.path.join(tempfile.gettempdir(), f"real_{uuid.uuid4()}_{real_sig_filename}")
        real_signature_file.save(real_sig_path)
        # Run verification
        result, error = verify_new_signature(model, real_sig_path, cropped_path)
        # Clean up temp files (optional)
        # os.remove(cropped_path)
        # os.remove(real_sig_path)
        # Show result on the same detect_signature page
        return render_template('detect_signature.html', image_data=cropped_image_data, candidates=None, uploaded_pdfs=uploaded_pdfs, verification_result=result, verification_error=error)
    return render_template('detect_signature.html', image_data=None, candidates=None, uploaded_pdfs=uploaded_pdfs, verification_result=None, verification_error='Missing input files.')

@app.route('/generate-report', methods=['POST'])
def generate_report():
    decision = request.form.get('decision', 'N/A')
    distance = request.form.get('distance', 'N/A')
    threshold = request.form.get('threshold', 'N/A')

    # Create a new PDF in memory
    pdf_buffer = BytesIO()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Paths to images
    logo_path = os.path.join('static', 'signisure_logo.png')
    signature_path = os.path.join('static', 'signisure_signature.png')

    # Add logo at the top
    if os.path.exists(logo_path):
        rect_logo = fitz.Rect(170, 30, 425, 130)
        page.insert_image(rect_logo, filename=logo_path)

    # Title
    page.insert_text((180, 150), "SigniSure Signature Verification Report", fontsize=18, fontname="helv", color=(0,0,0))

    # Main content
    y = 200
    page.insert_text((70, y), f"Prediction: {decision}", fontsize=15, fontname="helv", color=(0,0.5,0) if decision.lower()=='genuine' else (0.8,0,0))
    y += 30
    page.insert_text((70, y), f"Distance Score: {distance}", fontsize=13, fontname="helv", color=(0,0,0))
    y += 25
    page.insert_text((70, y), f"Threshold Used: {threshold}", fontsize=13, fontname="helv", color=(0,0,0))
    y += 40
    # Warnings and info
    warning_text = "Warning: This result is based on AI model predictions. For legal or official purposes, human verification is recommended.\n\nThe model may be sensitive to image quality, cropping, and signature clarity. Always double-check results in critical scenarios."
    page.insert_textbox(fitz.Rect(70, y, 525, y+80), warning_text, fontsize=11, fontname="helv", color=(0.7,0.2,0.2))
    y += 100
    # Footer text
    page.insert_text((70, 780), "Generated by SigniSure - AI-Powered Signature Verification", fontsize=10, fontname="helv", color=(0.3,0.3,0.3))
    # Add signature image at the bottom
    if os.path.exists(signature_path):
        rect_sig = fitz.Rect(200, 700, 400, 780)
        page.insert_image(rect_sig, filename=signature_path)

    doc.save(pdf_buffer)
    doc.close()
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name="signisure_report.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    # Load your model here
    model = load_model('best_signature_model.pth')
    app.run(debug=True, host='0.0.0.0', port=5000) 