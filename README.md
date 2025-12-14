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


# How the algorithm works (CNN vs simple Neural Networks)
## The problem:
Each MNIST image is a **28x28 grayscale** picture. The goal is to map that image to one of **10 classes** (digits from 0 to 9).
A naive approach is to flatten the image into a vector of **784 values** and feed it into a fully-connected neural network. While this can work, it ignored **2D structure** of the image(neighboring pixels, edges) and typically needs more parameters to learn the same visual patterns.

## Why CNNs are different (and better for images)
A **Convolutional Neural Network (CNN)** is designed to exploit the spatial structure of images.

Key ideas:
  - **Local receptive fields**: Convolution filters (kernels) look at small regions (for example **3x3**) instead of the whole image at once. This helps detect local patterns like edges and corners.
  - **Weight sharing**: The same filter is applied across the entire image. This dramatically reduces the number of parameters compared to fully-connected layers and allows the model to detect the same features anywhere in the image.
  - **Translation robustness**: Because features are detected regardless of location, CNNs are naturally more robust to small shifts in position (especially combined with pooling and augmentation).

### Layer-by-layer intuition
This model transforms the image through multiple stages:
1. **Convolution (Conv2D)**: Each convolution layer learns a set of filters that produce **feature maps**.
  - Early layers typically learn simple patterns (edges, curves).
  - Deeper layers combine these into more complex features (digit strokes and shapes).
2. **Batch normalization (BatchNorm)**: BatchNorm stabilizes training by normalizing activations, often improving convergence speed and overall accuracy.
3. **ReLU activation**: ReLU introduces non-linearity, allowing the network to model complex decision boundaries (instead of being limted to linear transformations).
4. **Max Pooling**: Pooling reduces spacial resolution (for example 26x26 to 13x13), making the model:
  - faster and more memory efficient
  - more robust to small translations and distortions
  - focused on the strongest activations (most important features)
5. **Flatten + Fully Connected Layers**:
  - After convolution + pooling, the feature maps represent high-level information.
  - Flatten converts them into a vector (in this case 36 x 5 x 5 = 900) and:
    - ```FC1``` learns a compact representation (900 -> 128)
    - ```Dropout(p=0.5)``` helps reduce overfitting by randomly disabling neurons during training
    - ```FC2``` outputs **10 logits**, one for each digit class
6. **Softmax (only for interpretation)**: During training we use ```CrossEntropyLoss```, which expects logits (no softmax needed in the model). In the GUI, we apply ```softmax``` to convert logits into probabilities for a confidence score.

 

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

I hope this explanation was enough to make an idea of how this model and these scripts actually work. Now have fun testing the model with your drawing talent :)
Thanks for accessing my repo! 
