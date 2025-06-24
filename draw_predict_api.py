# Instructions:
# 1. Start this backend: python draw_predict_api.py
# 2. In another terminal, run: python -m http.server 8000
# 3. Open http://localhost:8000/draw_predict_frontend.html in your browser

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import os
from scipy import ndimage
from dotenv import load_dotenv
import openai

# Model definition (must match training)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

MODEL_PATH = 'mnist_model.pth'
model = Net()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
else:
    model = None

# Load OpenAI key from .env
load_dotenv()
openai.api_key = os.getenv("openaikey")

def preprocess(image):
    # Convert to grayscale
    image = image.convert('L')
    # Invert so digit is white on black
    image = ImageOps.invert(image)
    arr = np.array(image)
    # Crop to bounding box
    nonzero = np.argwhere(arr > 20)  # threshold to ignore noise
    if nonzero.size:
        top_left = nonzero.min(axis=0)
        bottom_right = nonzero.max(axis=0)
        arr = arr[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    # Resize to 20x20
    img = Image.fromarray(arr)
    img = img.resize((20, 20), Image.LANCZOS)
    arr = np.array(img)
    # Place in 28x28 and center using center of mass
    new_img = np.zeros((28, 28), dtype=np.float32)
    # Compute center of mass
    cy, cx = ndimage.center_of_mass(arr)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = 10, 10  # fallback
    x0 = int(round(14 - cx))
    y0 = int(round(14 - cy))
    # Paste 20x20 into 28x28
    for y in range(20):
        for x in range(20):
            yy = y + y0
            xx = x + x0
            if 0 <= yy < 28 and 0 <= xx < 28:
                new_img[yy, xx] = arr[y, x]
    # Normalize to [0, 1]
    new_img = new_img / 255.0
    tensor = torch.tensor(new_img).view(1, 28*28)
    return tensor

def validate_with_gpt(image, model_prediction):
    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    prompt = (
        "This is a 28x28 pixel image of a handwritten digit. "
        "The model predicted it is a '{}'. "
        "Does this look correct? Reply with only the correct digit (0-9) or 'uncertain'."
    ).format(model_prediction)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a digit recognition expert."},
                {"role": "user", "content": prompt, "images": [{"image": img_str, "mime_type": "image/png"}]}
            ],
            max_tokens=5
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"OpenAI error: {e}"

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'Digit Prediction API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found. Please train and save mnist_model.pth.'}), 500
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    image_data = data['image']
    try:
        image_bytes = base64.b64decode(image_data.split(',')[-1])
        image = Image.open(io.BytesIO(image_bytes))
        tensor = preprocess(image)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        # Recreate the 28x28 image for GPT validation
        processed_img = Image.fromarray((tensor.view(28,28).numpy()*255).astype('uint8'))
        gpt_validation = validate_with_gpt(processed_img, pred)
        return jsonify({'prediction': int(pred), 'gpt_validation': gpt_validation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 