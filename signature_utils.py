from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import fitz  # PyMuPDF
from io import BytesIO
from model_loader import preprocess_signature, verify_signatures
import cv2

def preprocess_new_signature(image_path, target_size=105):
    """
    Preprocess a new signature image for your trained model
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale (essential for your model)
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply the same preprocessing as your validation transform
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),  # 105x105 to match training
            transforms.ToTensor(),                          # Convert to tensor [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize to [-1,1]
        ])
        
        processed_img = transform(img)
        
        # Add batch dimension for model input
        processed_img = processed_img.unsqueeze(0)  # Shape: [1, 1, 105, 105]
        
        return processed_img
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def verify_new_signature(model, reference_signature_path, test_signature_path, threshold=0.2602):
    """
    Verify a new signature against a reference signature
    """
    if model is None:
        return None, "Model not loaded"
    
    model.eval()
    
    # Preprocess both signatures
    ref_img = preprocess_new_signature(reference_signature_path)
    test_img = preprocess_new_signature(test_signature_path)
    
    if ref_img is None or test_img is None:
        return None, "Error preprocessing images"
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref_img = ref_img.to(device)
    test_img = test_img.to(device)
    
    # Get model predictions
    with torch.no_grad():
        ref_features, test_features = model(ref_img, test_img)
        distance = torch.nn.functional.pairwise_distance(ref_features, test_features).item()
    
    # Make decision
    is_genuine = distance <= threshold
    
    result = {
        'distance': distance,
        'threshold': threshold,
        'is_genuine': is_genuine,
        'decision': 'GENUINE' if is_genuine else 'FORGED'
    }
    
    return result, None

def extract_signature_from_pdf(pdf_path, bounding_box=None):
    """
    Extract signature from PDF using bounding box coordinates
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        if bounding_box:
            # Extract the specific area
            rect = fitz.Rect(bounding_box['x'], bounding_box['y'], 
                           bounding_box['x'] + bounding_box['width'], 
                           bounding_box['y'] + bounding_box['height'])
            pix = page.get_pixmap(clip=rect)
        else:
            # Get the whole page
            pix = page.get_pixmap()
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        doc.close()
        return img
        
    except Exception as e:
        print(f"Error extracting from PDF: {e}")
        return None

def load_model(model_path):
    """
    Load your trained signature verification model
    """
    try:
        # You'll need to implement this based on your model architecture
        # Example:
        # model = YourModelClass()
        # model.load_state_dict(torch.load(model_path, map_location=device))
        # model.to(device)
        # return model
        pass
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_signature_regions(image, min_area=500, max_area=50000):
    """Detect potential signature regions in document image"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    # Preprocessing for signature detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_candidates = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            extent = area / (w * h)
            if (1.5 <= aspect_ratio <= 8.0 and 0.1 <= extent <= 0.8):
                signature_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'confidence': calculate_signature_confidence(gray[y:y+h, x:x+w])
                })
    signature_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    return signature_candidates

def calculate_signature_confidence(signature_region):
    """Calculate confidence that a region contains a signature"""
    height, width = signature_region.shape
    binary = cv2.threshold(signature_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    stroke_density = np.sum(binary == 255) / (height * width)
    edges = cv2.Canny(signature_region, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    horizontal_projection = np.sum(binary, axis=0)
    horizontal_spread = np.std(horizontal_projection) / np.mean(horizontal_projection + 1)
    confidence = (stroke_density * 0.4 + edge_density * 0.3 + min(horizontal_spread, 1.0) * 0.3)
    return confidence 