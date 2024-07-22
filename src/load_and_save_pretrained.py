import torch
import torchvision

# Download the pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Save the model
torch.save(model.state_dict(), 'models/resnet50.pth')
