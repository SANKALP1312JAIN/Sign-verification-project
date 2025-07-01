def detect_signature_regions(image, min_area=500, max_area=50000):
    """Detect potential signature regions in document image"""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Preprocessing for signature detection
    # 1. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Morphological operations to connect signature strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    signature_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Signature characteristics
            aspect_ratio = w / h
            extent = area / (w * h)
            
            # Typical signature properties
            if (1.5 <= aspect_ratio <= 8.0 and  # Signatures are usually wider than tall
                0.1 <= extent <= 0.8):          # Reasonable fill ratio
                
                signature_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'confidence': calculate_signature_confidence(gray[y:y+h, x:x+w])
                })
    
    # Sort by confidence score
    signature_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    return signature_candidates

def calculate_signature_confidence(signature_region):
    """Calculate confidence that a region contains a signature"""
    
    # Features that indicate a signature
    height, width = signature_region.shape
    
    # 1. Stroke density
    binary = cv2.threshold(signature_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    stroke_density = np.sum(binary == 255) / (height * width)
    
    # 2. Edge complexity
    edges = cv2.Canny(signature_region, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # 3. Horizontal distribution (signatures spread horizontally)
    horizontal_projection = np.sum(binary, axis=0)
    horizontal_spread = np.std(horizontal_projection) / np.mean(horizontal_projection + 1)
    
    # Combine features into confidence score
    confidence = (stroke_density * 0.4 + 
                 edge_density * 0.3 + 
                 min(horizontal_spread, 1.0) * 0.3)
    
    return confidence
