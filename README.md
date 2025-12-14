# Handwritten Digit Recognition using CNN Neural Networks in PyTorch.
## Handwritten Digit Recognition is a classification problem, where the goal is to correctly identify digits [0-9] from images.
## Features of this project:
- CNN training on MNIST (train + test evaluation each epoch)
- Data augmentation during training (```RandomAffine:``` small rotations + translations)
- Automatically saves the best model checkpoint(```mnist_cnn_best.pth```)
- Learning rate scheduling with ```StepLR```
- Tkinter GUI for drawing digits and getting predictions + confidence

## To run the scripts:
### 1. Clone the repository:
```bash
  git clone https://github.com/AlexDuna/Handwritten_Digit_Recognition_NN.git
  cd CNN_Handwritten_Digit_Recognition
```
### 2. Install dependencies:
You can install the required libraries with:
```
  pip install torch torchvision pillow
```
- Also, if Tkinter is missing:
```
  sudo apt-get install python3-tk
```

### 3. Train the model:
```
  python ./train_mnist_dataset.py
```

This will create a file that saves the best model (with the highest testing accuracy):
```
  mnist_cnn_best.pth
```

### 4. Run the GUI:
```
  python ./draw_gui.py
```
- After that, you can draw, and get predictions based on your drawings :)

- The GUI should look like this:
- Default:
<img src="/images/Default.png" alt="isolated"/>

- Prediction 1:
<img src="/images/Prediction1.png" alt="isolated"/>

- Prediction 2:
<img src="/images/Prediction2.png" alt="isolated"/>



# Model Architecture
<img src="/images/Representation.png" alt="isolated"/>

### Examples of MNIST dataset:
<img src="/images/MNIST_dataset.png" alt="isolated"/>

The CNN consists of:
  - **Conv2D(1 -> 24, kernel=3)** + BatchNorm + ReLU + MaxPool(2)
  - **Conv2D(24 -> 36, kernel=3)** + BatchNorm + ReLU + MaxPool(2)
  - Flatten
  - **Linear(36 * 5 * 5 -> 128)** + ReLU + Dropout(p=0.5)
  - **Linear(128 -> 10)** (digits 0-9)

- Input shape: **(N,1,28,28)**
- Output: logits for **10 classes**

## Training Details
- Dataset: ```torchvision.dataset.MNIST```
- Train-time augmentation:
  - random rotation up to +/- 15 degrees
  - random translation up to 10% on x/y
- Optimizer: **SGD** with momentum and weight decay
  - ```lr = 0.01```, ```momentum = 0.9```, ```weight-decay = 1e-4```
- Loss: ```CrossEntropyLoss```
- LR Scheduler: ```StepLR(step_size = 10, gamma = 0.1)```
- Epochs: ```20```
- The script saves the best model (highest test accuracy) to: ```mnist_cnn_best.pth```

## How the GUI preprocessing works:
The GUI converts your drawings into a MNIST-like image:
1. Finds the bounding box of the drawn digit (non-black pixels)
2. Crops the digit region
3. Resizes it so the longest side becomes ~20 pixels (keeps aspect ratio)
4. Centers it inside a **28x28** black image
5. Converts to tensor and runs inference

The GUI outputs:
  - **Predicted digit**
  - **Confidence score** (softmax probability)

