from torchvision import transforms
from torchvision.transforms import RandAugment


simple_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transform1 = transforms.Compose([
        # Resize images to 224x224
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.3), scale=(0.5, 1)),
        transforms.RandomRotation(degrees=(-20, 20)),
        # transforms.Grayscale(3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        transforms.RandomInvert(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

# Define the second argumentation technique then compare to the first one
transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

transform3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Add RandAugment with N, M(hyperparameter)
transform3.transforms.insert(0, RandAugment(4, 9))

transform_efficientNetB2 = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_efficientNet_V2_S = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_resnet34 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
