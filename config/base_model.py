import torchvision
import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define EfficientNet with different base model
weight_v2_EfficientNet_M = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
base_v2_modelEfficientNet_M = torchvision.models.efficientnet_v2_m(weights=weight_v2_EfficientNet_M).to(device)

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

weightResNet50 = torchvision.models.ResNet50_Weights.DEFAULT
base_modelResNet50 = torchvision.models.resnet50(weights=weightResNet50).to(device)

weightResNet101 = torchvision.models.ResNet101_Weights.DEFAULT
base_modelResNet101 = torchvision.models.resnet101(weights=weightResNet101).to(device)

weight_tiny_ConvNeXt = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
tiny_modelWeightConvNeXt = torchvision.models.convnext_tiny(weights=weight_tiny_ConvNeXt).to(device)
