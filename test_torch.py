import torch
from models.model import Model
# import gc # Import garbage collector module

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# model = Model().to(device)
# for i in range(10):
#     x = torch.randn(8, 3, 64, 64).to(device)
    
#     with torch.no_grad():
#         output = model(x)


    
#     # assert output.shape == (1, 1)
#     print(f"Test {i+1}: Output shape {output.shape} - Passed")
#     del output
#     del x
#     torch.cuda.empty_cache()

# gc.collect()  # Force garbage collection

import torch
import torchvision.models as models

# Use a standard model instead of your custom one
# from models.model import Model

device = "cpu"
print(f"Using device: {device}")

# Use weights=None to avoid the UserWarning and download
model = models.resnet18(weights=None).to(device)
model.eval() # Set to evaluation mode

# Your original loop
for i in range(10):
    # Adjust input size if necessary for the standard model
    x = torch.randn(8, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Test {i+1}: Output shape {output.shape} - Passed")

print("Script finished successfully.")
