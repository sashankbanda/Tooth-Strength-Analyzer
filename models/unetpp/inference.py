import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from .model import get_model_structural_segmentation

class ToothStructureSegmentor:
    def __init__(self, weights_path=None, device=None, input_size=(320, 320), encoder_weights="imagenet"):
        self.device = device if device else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = 4 # Background, Root, Bone, CEJ
        self.input_size = input_size
        self.model = get_model_structural_segmentation(num_classes=self.num_classes, encoder_weights=encoder_weights)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.input_size), # Resize to U-Net++ expected input
        ])

    def predict(self, image):
        """
        Args:
            image: PIL Image or numpy array (RGB) - The cropped tooth image.
            
        Returns:
            Segmentation mask (numpy array) of shape (H, W) with values 0-3.
        """
        original_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        img_tensor = self.transform(image).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            
        # Resize mask back to original ROI size
        # Use nearest neighbor to keep class labels
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred_mask_resized
