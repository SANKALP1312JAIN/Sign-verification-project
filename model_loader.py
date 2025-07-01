import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

def get_validation_transform():
    return transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

class FinalSiameseNet(nn.Module):
    def __init__(self, dropout_p=0.4):
        super(FinalSiameseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_once(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return nn.functional.normalize(x, p=2, dim=1)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

def load_model(model_path=None, dropout_p=0.4):
    """
    Load your trained signature verification model
    
    Args:
        model_path (str): Path to your trained model weights
        
    Returns:
        model: Loaded PyTorch model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = FinalSiameseNet(dropout_p=dropout_p)
    
    if model_path and os.path.exists(model_path):
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        # Initialize with random weights (for demo purposes)
        print("No model weights found. Using randomly initialized model.")
    
    model.to(device)
    model.eval()
    
    return model

def preprocess_signature(image_path, target_size=105):
    """
    Preprocess signature image for model input
    
    Args:
        image_path (str): Path to signature image
        target_size (int): Target size for resizing
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply preprocessing
        transform = get_validation_transform()
        
        processed_img = transform(img)
        processed_img = processed_img.unsqueeze(0)  # Add batch dimension
        
        return processed_img
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def verify_signatures(model, genuine_path, test_path, threshold=0.28):
    """
    Verify signatures using the loaded model
    
    Args:
        model: Loaded PyTorch model
        genuine_path (str): Path to genuine signature
        test_path (str): Path to test signature
        threshold (float): Distance threshold for verification
        
    Returns:
        dict: Verification results
    """
    if model is None:
        return None, "Model not loaded"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess signatures
    genuine_tensor = preprocess_signature(genuine_path)
    test_tensor = preprocess_signature(test_path)
    
    if genuine_tensor is None or test_tensor is None:
        return None, "Error preprocessing images"
    
    # Move to device
    genuine_tensor = genuine_tensor.to(device)
    test_tensor = test_tensor.to(device)
    
    # Get model predictions
    with torch.no_grad():
        genuine_embedding, test_embedding = model(genuine_tensor, test_tensor)
        distance = torch.nn.functional.pairwise_distance(genuine_embedding, test_embedding).item()
    
    # Make decision
    is_genuine = distance <= threshold
    
    result = {
        'distance': distance,
        'threshold': threshold,
        'is_genuine': is_genuine,
        'decision': 'GENUINE' if is_genuine else 'FORGED'
    }
    
    return result, None

# Example usage:
if __name__ == "__main__":
    # Load model
    model = load_model()
    
    # Example verification
    # result, error = verify_signatures(model, "genuine_signature.png", "test_signature.png")
    # if result:
    #     print(f"Result: {result['decision']}")
    #     print(f"Distance: {result['distance']:.4f}")
    #     print(f"Confidence: {result['confidence']:.4f}")
    # else:
    #     print(f"Error: {error}") 