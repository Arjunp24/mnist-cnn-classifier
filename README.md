# MNIST CNN Classifier

[![Model Tests](https://github.com/arjunp24/mnist-cnn-classifier/actions/workflows/model_tests.yml/badge.svg)](https://github.com/arjunp24/mnist-cnn-classifier/actions/workflows/model_tests.yml)

A PyTorch implementation of a CNN model for MNIST digit classification that achieves >95% accuracy in one epoch with less than 25,000 parameters.

## Project Structure 

```
mnist-cnn
│   README.md
│   requirements.txt
│   train.py
│   test_model.py
│   model.py
│   mnist_cnn.pth
|   .gitignore
└───.github
│   └───workflows
│       │   model_tests.yml 
└───augmented_samples
```


## Model Architecture

- 3 convolutional blocks with batch normalization
- Each block: Conv2D -> BatchNorm -> ReLU -> MaxPool
- Total parameters: ~23K
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- matplotlib
- tqdm

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/arjunp24/mnist-cnn.git
    cd mnist-cnn
    ```

2. Create and activate virtual environment:

    Create virtual environment
    ```
    python -m venv mnist_env
    ```
    Activate on Windows
    ```
    mnist_env\Scripts\activate
    ```
    Activate on Unix/MacOS
    ```
    source mnist_env/bin/activate
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

Run the training script:
    ```
    python train.py
    ```

This will:
- Download MNIST dataset (if not present)
- Save augmented sample images
- Train the model for one epoch
- Save the trained model as 'mnist_cnn.pth'

### Running Tests Locally

Run all tests:
    ```
    pytest test_model.py -v
    ```

Run specific test:
    ```
    pytest test_model.py::test_model_accuracy -v
    ```

### GitHub Actions

The project includes automated testing through GitHub Actions:
1. Push your changes to GitHub
2. Go to Actions tab
3. Check test results

Tests verify:
- Model has <25K parameters
- Achieves >95% accuracy
- Correct input/output shapes
- Proper error handling
- Valid output ranges

## Model Features

- Data augmentation (rotation, resizing)
- Batch normalization for training stability
- Image normalization (mean=0, std=1)
- GPU support when available
- Progress bar during training

## Notes

- The model is designed to achieve >95% accuracy in a single epoch
- Training uses Adam optimizer with learning rate 0.0034
- Batch size is set to 16 for better training stability
- Data augmentation helps prevent overfitting

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
