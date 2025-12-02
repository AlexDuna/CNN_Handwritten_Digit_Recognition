import tkinter as tk
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image, ImageGrab, ImageDraw

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional part
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=24, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels=36, kernel_size=3)

        # Fully connected part
        self.fc1 = nn.Linear(36 * 5 * 5, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x,1) # flatten all dimensions except batch

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# load trained model
device = torch.device("cpu") # cpu is good enough for GUI
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_best.pth", map_location=device))
model.eval()

# Transforms for GUI image -> tensor 1x1x28x28
transform = transforms.Compose([
    transforms.ToTensor()
])

CANVAS_SIZE = 280
BRUSH_SIZE = 8

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black"
        )
        self.canvas.pack(padx=10,pady=10)

        # internal PIL image on which we draw in parallel
        self.image = Image.new("L", (CANVAS_SIZE,CANVAS_SIZE), color = 0)  #black
        self.draw_image = ImageDraw.Draw(self.image) 

        #draw with left mouse button
        self.canvas.bind("<B1-Motion>", self.draw)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.predict_btn = tk.Button(btn_frame, text="Predict", command = self.predict)
        self.predict_btn.pack(side=tk.LEFT,padx=5)

        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE

        # draw on canvas (UI)
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")

        # draw same thing on PIL image also
        self.draw_image.ellipse((x-r,y-r, x+r,y+r), fill=255) # 255 = white on black background

    def clear(self):
        self.canvas.delete("all")
        self.result_label.config(text = "Draw a digit (0-9)")

        # reset internal image to black
        self.image = Image.new("L", (CANVAS_SIZE,CANVAS_SIZE), color = 0)
        self.draw_image = ImageDraw.Draw(self.image)

    def predict(self):
        # internal image, white digit on black background
        img = self.image

        # find digit bounding box (zone where pixels != 0)
        bound_box = img.getbbox()
        if bound_box is None:
            self.result_label.config(text = "No digit detected")
            return

        # 1. crop on digit zone
        img = img.crop(bound_box)

        # 2. resize while keeping aspect ratio so "1" stays "1"
        box_w, box_h = img.size
        # scale so that the longest side becomes 20 pixels
        scale = 20.0 / max(box_w, box_h)
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        img = img.resize((new_w, new_h))

        # 3. create a black 28x28 img and paste the 20x20 img on it
        new_img = Image.new("L", (28,28), color=0)
        upper_left = ((28-new_w) // 2, (28-new_h) // 2)
        new_img.paste(img,upper_left)

        # 4. transform to tensor [1,1,28,28]
        tensor_img = transform(new_img)
        img_t = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            digit = predicted.item()
            confidence = probs[0, digit].item()

        self.result_label.config(text = f"Predicted digit: {digit} (Confidence: {confidence: .2f})")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
        
