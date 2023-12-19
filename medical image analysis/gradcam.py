import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
from torchcam.methods import GradCAM
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the model
model = densenet121(pretrained=True)
model.eval()

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open and preprocess the image
input_image = Image.open("/home/shaijal/ProtoTree/data/xrays/xrays_train/Cardiomegaly/patient00069_study45_view1_frontal.jpg").convert("RGB")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# If you have a GPU, put everything on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

# Grad-CAM
cam_extractor = GradCAM(model=model, target_layer=model.features[-1])
scores = model(input_batch)
cam = cam_extractor(scores)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                         subplot_kw={'xticks': [], 'yticks': []})

# Plot original image
axes[0].imshow(input_image)
axes[0].set_title('Input Image')

# Plot image with Grad-CAM
axes[1].imshow(input_image)
axes[1].imshow(cam.cpu().numpy(), cmap='jet', alpha=0.5)
axes[1].set_title('Grad-CAM')

plt.show()
