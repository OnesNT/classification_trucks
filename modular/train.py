import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import RandAugment
from ultralytics import YOLO
from modular.data_setup import TruckDataset
# import modular.engine
# import modular.model_builder
# import modular.utils
# import modular.evaluate
from modular import engine
from modular import model_builder
from modular import utils
import cv2
import torch.nn.functional as F
import matplotlib.image as mpimg
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from config import base_model, setup_hyperparameters, transforms


# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Freezing layers with ratios
# freeze_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
freeze_ratios = [0]

# create model, optimizer, loss function for model
def create_model(model_choice, base_model, schedule_lr):
    # output size
    output_shape = len(TruckDataset.classes)

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if model_choice == 'e':
        model = model_builder.EfficientNetCustom(base_model, output_shape).to(device)
        print("Create model with EfficientNet")
    elif model_choice in ['r34', 'r50']:
        model = model_builder.ResNetCustom(base_model, output_shape).to(device)
        print("Create model with ResNet")
    elif model_choice == 'c':
        model = model_builder.ConvNeXtCustom(base_model, output_shape).to(device)
        print("Create model with ConvNeXt")
    elif model_choice == 'e_v2':
        model = model_builder.EfficientNetV2Custom(base_model, output_shape).to(device)
        print('Create model with EfficientNetv2')
    else:
        print('invalid model choice')
        return None, None, None, None


    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    optimizer = optim.SGD(model.parameters(), lr=setup_hyperparameters.LEARNING_RATE)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    if schedule_lr:
        return model, optimizer, loss_fn, scheduler
    else:
        return model, optimizer, loss_fn, None


# Load detector (yolov8x) for detect truck
def detector(detector_path):
    # detector_path = '../weights/yolov8m/7/best.pt'
    model = YOLO(detector_path)
    return model


# Load classifier from which trained
def classifier(model_path, model_choice, base_model, schedule_lr):
    # model_path = 'models/model3.pth'
    model, optimizer, loss_fn, scheduler = create_model(model_choice, base_model, schedule_lr)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model



# count_max_1 = 0
# count_max_2 = 0
#
#
# def threshold_check(probs, pics_dir, model_path, base_model):
#
#     os.makedirs(f"/home/user/NgoQuang/datasets/threshold_below_{probs}/truck_1", exist_ok=True)
#     os.makedirs(f"/home/user/NgoQuang/datasets/threshold_below_{probs}/truck_2", exist_ok=True)
#     des = f"/home/user/NgoQuang/datasets/threshold_below_{probs}"
#     src = "/home/user/NgoQuang/classification_truck_datasets"
#
#     model = classifier(model_path, base_model)
#     model.to(device)
#
#     subdirs = ['truck_1', 'truck_2']
#
#     for subdir in subdirs:
#         pics_path = os.path.join(pics_dir, subdir)
#         pics = os.listdir(pics_path)
#
#         for pic in pics:
#             # Build the full path to the image
#             img_path = os.path.join(pics_path, pic)
#
#             # Load the image using OpenCV or similar
#             image = cv2.imread(img_path)
#
#             if image is None:
#                 print(f"Failed to load image: {img_path}")
#                 continue
#
#             image = cv2.resize(image, (224, 224))
#
#             # Convert the image to a tensor and move to the specified device
#             transform = transforms.ToTensor()
#             data = transform(image).unsqueeze(0).to(device)
#
#             # Perform classification
#             logits = model(data)
#
#             # Convert logits to probabilities
#             probabilities = F.softmax(logits, dim=1)[0]
#
#             # Check if both class probabilities are below the threshold
#             if max(probabilities) < probs:
#                 # print(f"Image {img_path} has low confidence: {max(probabilities)}")
#                 shutil.copy2(os.path.join(pics_path, pic), os.path.join(des, subdir, pic))
#                 # print(os.path.join(pics_path, pic), os.path.join(des, subdir, pic))
#                 # print(os.path.join(des, subdir, pic))
#
#     return 0


def detect_and_classify_truck(img_dir, output_dir, model_path, base_model, detector_path, schedule_lr):

    # numerical1 = 0
    # numerical2 = 0
    # numerical3 = 0

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, "truck_1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "truck_2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cropped'), exist_ok=True)

    # Load the models
    model = classifier(model_path, base_model, schedule_lr)
    model.to(device)

    detector_model = detector(detector_path)
    detector_model.to(device)

    for count, img_file in enumerate(os.listdir(img_dir)):

        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Detect objects in the image
        results = detector_model(image, verbose=False)
        print(model.names)

        # Iterate over detected objects
        for i, box in enumerate(results[0].boxes):

            if box is None or box.xyxy is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            # Check if the detected object is a truck
            if class_id == 0:
                # print(f"Detected class ID: {class_id}")

                cropped_img = image[y1:y2, x1:x2]
                # numerical3 += 1
                save_path = os.path.join(output_dir, 'cropped', img_file)
                cv2.imwrite(save_path, cropped_img)

                transform = transforms.ToTensor()
                cropped_img_tensor = transform(cropped_img).unsqueeze(0).to(device)

                # Perform classification
                logits = model(cropped_img_tensor)
                probabilities = F.softmax(logits, dim=1)[0]

                if probabilities[0] >= probabilities[1]:
                    # numerical1 += 1
                    # save_path = os.path.join(output_dir, 'truck_1',
                    #                          f'img{numerical1}_{results[0].names[0]}_{x1, y1, x2, y2}.jpg')
                    save_path = os.path.join(output_dir, 'truck_1', img_file)
                else:
                    # numerical2 += 1
                    # save_path = os.path.join(output_dir, 'truck_2',
                    #                          f'img{numerical2}_{results[0].names[0]}_{x1, y1, x2, y2}.jpg')
                    save_path = os.path.join(output_dir, 'truck_2', img_file)
                # Save the original image with annotation
                cv2.imwrite(save_path, image)


def inference_for_truck_1(img_dir, out_dir, model_path, base_model, model_choice, schedule_lr):
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)

    inf_dir_out = os.path.join(out_dir, model_path.split('/')[-1].split('.')[0])
    if os.path.exists(inf_dir_out):
        shutil.rmtree(inf_dir_out)

    os.makedirs(os.path.join(inf_dir_out, "truck_1"), exist_ok=True)
    os.makedirs(os.path.join(inf_dir_out, "truck_2"), exist_ok=True)
    #
    # # Load the model
    model = classifier(model_path, model_choice, base_model, schedule_lr)
    model.to(device)

    # Define image transformation
    transform = transforms.simple_transform

    for count, img_file in enumerate(os.listdir(img_dir)):
        # Get image from img_dir
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_file}. Skipping.")
            continue

        PIL_image = Image.fromarray(image)

        # Transform image for classification
        img_file_tensor = transform(PIL_image).unsqueeze(0).to(device)

        # Perform classification
        with torch.no_grad():
            logits = model(img_file_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        if probabilities[0] >= probabilities[1]:
            save_path = os.path.join(inf_dir_out, 'truck_1', f'{probabilities[0].item():.4f}_{img_file}')
        else:
            save_path = os.path.join(inf_dir_out, 'truck_2', f'{probabilities[1].item():.4f}_{img_file}')

        # Save the original image with annotation
        cv2.imwrite(save_path, image)


# Define transformations including data augmentation technique
def data_set_up(transform):

    root_dir = '/home/user/Quang/classification_truck_datasets'
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    validation_dir = os.path.join(root_dir, 'val')

    train_dataset = TruckDataset(root_dir=train_dir, transform=transform)
    test_dataset = TruckDataset(root_dir=test_dir, transform=transforms.simple_transform)
    validation_dataset = TruckDataset(root_dir=validation_dir, transform=transforms.simple_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=setup_hyperparameters.BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=setup_hyperparameters.BATCH_SIZE, shuffle=False, num_workers=8)
    validation_dataloader = DataLoader(validation_dataset, batch_size=setup_hyperparameters.BATCH_SIZE, shuffle=False, num_workers=8)

    return train_dataloader, test_dataloader, validation_dataloader


# Train model from base model and how we use argumentation technique
def train_model(transform, model_choice, base_model, version_model, schedule_lr):
    min_loss = 999
    results_best = dict()
    ratio_best = 0

    for ratio in freeze_ratios:
        model, optimizer, loss_fn, scheduler = create_model(model_choice, base_model, schedule_lr)
        if model is None or optimizer is None or loss_fn is None:
            print("Failed to create model, optimizer, or loss function. Exiting training.")
            return None
        model.to(device)

        train_dataloader, test_dataloader, validation_dataloader = data_set_up(transform)

        # Set the freeze ratio
        print(f"With ratio: {ratio}")
        model.freeze_ratio(ratio)

        # Start training with help from engine.py
        results = engine.train( model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                validation_dataloader=validation_dataloader,
                                epochs=setup_hyperparameters.NUM_EPOCHS,
                                device=device,
                                schedule_lr=schedule_lr,
                                scheduler=scheduler
                                )

        test_loss_min_ratio = min(results["test_loss"])
        if min_loss > test_loss_min_ratio:
            model_best = model
            min_loss = test_loss_min_ratio
            results_best = results
            # best_epoch = NUM_EPOCHS
            ratio_best = ratio

    model_path = '/home/user/Quang/truck_classification/models'
    get_models = os.listdir(model_path)
    available_num = 0

    for model in get_models:
        s = model.split(".")
        if s[0].replace("model", ""):
            numerical_order = int(s[0].replace("model", ""))
            available_num = max(available_num, numerical_order)

    # Save the model with help from utils.py
    utils.save_model(model=model_best,
                     optimizer=optimizer,
                     best_ratio=ratio_best,
                     accuracy_test=results_best['test_acc'],
                     loss_test=results_best['test_loss'],
                     accuracy_train=results_best['train_acc'],
                     loss_train=results_best['train_loss'],
                     accuracy_val=results_best['val_acc'],
                     loss_val=results_best['val_loss'],
                     target_dir="models",
                     model_name=f"model{available_num+1}.pth",
                     transform=transform,
                     model_choice=model_choice,
                     lrs=results_best["lrs"][0],
                     version_model=version_model)

    return results_best

