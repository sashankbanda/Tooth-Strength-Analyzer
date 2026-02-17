from fastapi.testclient import TestClient
from api.main import app
import numpy as np
import cv2
import io
from PIL import Image

client = TestClient(app)

def create_dummy_image():
    # Create a 200x100 white image
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    # Add a black box to simulate something?
    # The models are untrained so output will be garbage, but specific shapes might trigger "No root detected".
    # We just want to test pipeline connectivity (200 OK).
    img_pil = Image.fromarray(img)
    return img_pil

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "tooth-strength-analysis"}

def test_analyze_endpoint():
    # Create dummy image bytes
    img = create_dummy_image()
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {'scan': ('test.png', img_byte_arr, 'image/png')}
    
    # This might fail if models crash on random input or take too long, 
    # but we are testing connectivity.
    # The untrainted Mask R-CNN might optimize to output 0 boxes.
    # If 0 boxes, the API returns empty list of teeth.
    response = client.post("/api/v1/analyze", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "scan_id" in data
    assert "teeth" in data
