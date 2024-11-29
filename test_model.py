import torch
from model import SimpleCNN
import pytest

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25000"

def test_model_accuracy():
    # Load the trained model
    model = SimpleCNN()
    try:
        # Load weights with map_location to handle GPU/CPU compatibility
        checkpoint = torch.load('mnist_cnn.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        final_accuracy = checkpoint['final_accuracy']
        print(f"Loaded model with accuracy: {final_accuracy:.2f}%")
        
        # Check if accuracy meets the requirement
        assert final_accuracy >= 95.0, f"Model accuracy {final_accuracy}% is below the required 95%"
        
    except AssertionError as e:
        pytest.fail(str(e))
    except Exception as e:
        pytest.fail(f"Could not load trained model weights: {str(e)}")

def test_model_output_shape():
    model = SimpleCNN()
    batch_size = 32
    # Create dummy input (batch_size, channels, height, width)
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    # Check output shape is correct (batch_size, num_classes)
    assert output.shape == (batch_size, 10), f"Expected output shape {(batch_size, 10)}, got {output.shape}"

def test_model_input_validation():
    model = SimpleCNN()
    # Test with wrong input dimensions
    with pytest.raises(RuntimeError):
        wrong_input = torch.randn(1, 3, 28, 28)  # Wrong number of channels
        model(wrong_input)

def test_model_output_values():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    # Check if output contains valid probabilities
    assert torch.all(output >= -100) and torch.all(output <= 100), "Output values are not in a reasonable range"
    assert not torch.any(torch.isnan(output)), "Output contains NaN values"