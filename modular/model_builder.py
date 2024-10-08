import torch
import torchvision
import torch.nn as nn
from modular.data_setup import TruckDataset


class EfficientNetV2Custom(nn.Module):
    def __init__(self, base_model, output_shape):
        super(EfficientNetV2Custom, self).__init__()
        self.base_model = base_model

        # print(self.base_model)
        num_features = self.base_model.classifier[1].in_features

        print(num_features)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=num_features, out_features=output_shape, bias=True)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_ratio(self, ratio):
        for idx, param in enumerate(self.parameters()):
            if idx > int(ratio * len(list(self.parameters()))):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return 0


# Custom EfficientNet model
class EfficientNetCustom(nn.Module):
    def __init__(self, base_model, output_shape):
        super(EfficientNetCustom, self).__init__()
        self.base_model = base_model

        num_features = self.base_model.classifier[1].in_features
        print(num_features)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=num_features, out_features=output_shape, bias=True)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_ratio(self, ratio):
        for idx, param in enumerate(self.parameters()):
            if idx > int(ratio * len(list(self.parameters()))):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return 0


class ResNetCustom(nn.Module):
    def __init__(self, base_model, output_shape):
        super(ResNetCustom, self).__init__()
        self.base_model = base_model

        num_features = self.base_model.fc.in_features
        print(num_features)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=num_features, out_features=output_shape, bias=True)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_ratio(self, ratio):
        for idx, param in enumerate(self.parameters()):
            if idx > int(ratio * len(list(self.parameters()))):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return 0


class ConvNeXtCustom(nn.Module):
    def __init__(self, base_model, output_shape):
        super(ConvNeXtCustom, self).__init__()
        self.base_model = base_model

        num_features = self.base_model.classifier[2].in_features

        print(num_features)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=num_features, out_features=output_shape, bias=True)
        )

    def forward(self, x):
        return self.base_model(x)

    def freeze_ratio(self, ratio):
        for idx, param in enumerate(self.parameters()):
            if idx > int(ratio * len(list(self.parameters()))):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return 0