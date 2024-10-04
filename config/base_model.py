import torchvision
import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define EfficientNet with different base model
weight_v2_EfficientNet_S = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
base_v2_modelEfficientNet_S = torchvision.models.efficientnet_v2_s(weights=weight_v2_EfficientNet_S).to(device)

weights2 = torchvision.models.EfficientNet_B2_Weights.DEFAULT
base_model2 = torchvision.models.efficientnet_b2(weights=weights2).to(device)

weights1 = torchvision.models.EfficientNet_B1_Weights.DEFAULT
base_model1 = torchvision.models.efficientnet_b1(weights=weights1).to(device)

weights0 = torchvision.models.EfficientNet_B0_Weights.DEFAULT
base_model0 = torchvision.models.efficientnet_b0(weights=weights0).to(device)

weightResNet34 = torchvision.models.ResNet34_Weights.DEFAULT
base_modelResNet34 = torchvision.models.resnet34(weights=weightResNet34).to(device)

weightConvNeXt = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
base_modelWeightConvNeXt = torchvision.models.convnext_small(weights=weightConvNeXt).to(device)
