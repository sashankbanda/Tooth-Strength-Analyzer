import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from .model import get_model_instance_segmentation

class ToothInstanceSegmentor:
    def __init__(self, weights_path=None, device=None, pretrained=True):
        self.device = device if device else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = 2 # Background + Tooth
        self.model = get_model_instance_segmentation(self.num_classes, pretrained=pretrained)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def predict(self, image, confidence_threshold=0.85):
        """
        Args:
            image: PIL Image or numpy array (RGB)
            confidence_threshold: Minimum score to keep a prediction
            
        Returns:
            List of dictionaries containing 'box', 'mask', 'score', 'label' for each detected tooth.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        img_tensor = self.transform(image).to(self.device)
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            prediction = self.model(img_tensor)[0]

        results = []
        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].item()
            if score >= confidence_threshold:
                mask = prediction['masks'][i, 0].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                
                results.append({
                    'box': box, # [x1, y1, x2, y2]
                    'mask': mask,
                    'score': score,
                    'label': prediction['labels'][i].item() # Should be 1 for Tooth
                })
        
        return results
