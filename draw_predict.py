import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn

# Model definition (must match the training script)
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

# Load trained model weights
MODEL_PATH = 'mnist_model.pth'  # You need to save your trained model to this file after training
model = Net()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def preprocess(image):
    # Convert to grayscale, resize, invert, normalize, and convert to tensor
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.LANCZOS)
    arr = np.array(image) / 255.0
    arr = arr.astype(np.float32)
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 28, 28)
    tensor = tensor.view(1, 28*28)
    return tensor

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Draw a Digit (MNIST)')
        self.canvas_size = 280
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()
        self.button = tk.Button(self, text='Predict', command=self.predict)
        self.button.pack()
        self.clear_btn = tk.Button(self, text='Clear', command=self.clear)
        self.clear_btn.pack()
        self.image = Image.new('RGB', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.last_x = None
        self.last_y = None
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=16, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill='black', width=16)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill='white')

    def predict(self):
        tensor = preprocess(self.image)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        messagebox.showinfo('Prediction', f'The drawn digit is: {pred}')

if __name__ == '__main__':
    app = App()
    app.mainloop() 