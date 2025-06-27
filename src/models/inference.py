import torch

def predict_with_pretrained_dnn(model, X, test_complexes):
    """Make predictions using a pretrained DNN model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Move data to the same device as the model
    if hasattr(X, 'to'):
        X = X.to(device)
    
    with torch.no_grad():
        predictions = model(X, test_complexes)
    
    return predictions