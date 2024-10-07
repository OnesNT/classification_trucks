import os
import cv2
import torch
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from IPython.display import display
import shutil
import random
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from modular.train import data_set_up, classifier, create_model
from PIL import Image
# import gc
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def save_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               best_ratio: int,
               accuracy_test: list,
               loss_test: list,
               accuracy_train: list,
               loss_train: list,
               target_dir: str,
               model_name: str,
               transform, model_choice, version_model, lrs):
    """Saves a PyTorch model along with its state, optimizer state, epoch, and loss.

    Args:
        model: A target PyTorch model to save.
        optimizer: The optimizer for the model.
        best_ratio: The ratio that model have the highest efficient.
        accuracy: The list accuracy in best model.
        loss: The list loss in best model value.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
                    either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict along with additional information
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({
        'best_ratio': best_ratio,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_test': loss_test,
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'loss_train': loss_train,
        'transform': transform,
        'model_choice': model_choice,
        'version_model': version_model,
        'lrs': lrs,


    }, model_save_path)


def load_model(model_path, model_choice, base_model, schedule_lr):
    """Loads a PyTorch model along with its state, optimizer state, epoch, and loss.

    Args:
        model_path: The model's path to load the state dictionary into.

    Returns:
        epoch: The epoch at which the model was saved.
        loss: The loss value at the time of saving.
    """

    model, optimizer, loss_fn, scheduler = create_model(model_choice, base_model, schedule_lr)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_ratio = checkpoint['best_ratio']
    loss_train = checkpoint['loss_train']
    loss_test = checkpoint['loss_test']
    accuracy_test = checkpoint['accuracy_test']
    accuracy_train = checkpoint['accuracy_train']
    transform = checkpoint['transform']
    model_choice = checkpoint['model_choice']
    version_model = checkpoint['version_model']
    lrs = checkpoint['lrs']
    # print(list(checkpoint.keys()))
    model.eval()

    print(f'model_choice: {model_choice}, version_model: {version_model}, best_ratio: {best_ratio}')
    print(f'loss_train: {loss_train}, accuracy_train: {accuracy_train}')
    print(f'loss_test: {loss_test}, accuracy_test: {accuracy_test}')
    print(f'lrs: {lrs}')
    print(f'transform: {str(transform)}')

    return {'model_choice': model_choice,
            "best_ratio": best_ratio,
            "loss_train": loss_train,
            "loss_test": loss_test,
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "version_model": version_model,
            "transform": transform,
            "lrs": lrs}


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluates the model on the provided dataloader and calculates accuracy.

    Args:
        model: The model to evaluate.
        dataloader: The dataloader with test data.
        device: The device to use for computation ('cuda' or 'cpu').

    Returns:
        accuracy: The accuracy of the model on the test dataset.
    """
    model.to(device)
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Make predictions on the test data using the trained model
            y_test_pred = model(X_test)

            # Round the predictions to the nearest integer and convert to torch.int dtype
            y_test_pred = y_test_pred.round().to(dtype=torch.int)

            # Flatten the tensors to 1D arrays
            all_true_labels.append(y_test.view(-1).cpu().numpy())
            all_pred_labels.append(y_test_pred.view(-1).cpu().numpy())

    # Concatenate all batches to form the final arrays
    true_labels = np.concatenate(all_true_labels, axis=0)
    predicted_labels = np.concatenate(all_pred_labels, axis=0)

    # Calculate accuracy
    num_correct = (predicted_labels == true_labels).sum()
    total_samples = len(true_labels)
    accuracy = num_correct / total_samples

    return accuracy


def check_transformed_image(img_folder, transform):
    # Create output directory if it doesn't exist
    save_dir = "/home/user/Quang/datasets/transformed_image"
    name_folder = img_folder.split('/')[-1]
    save_path_folder = os.path.join(save_dir, name_folder)

    os.makedirs(save_path_folder, exist_ok=True)

    # Remove existing folder if it exists
    if name_folder in os.listdir(save_dir):
        shutil.rmtree(save_path_folder)
        os.makedirs(save_path_folder, exist_ok=True)

    # Process each image in the folder
    for img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping")
            continue

        # Convert the image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply the transformation
        transformed_image = transform(image_pil)

        # # Convert the transformed image tensor to a NumPy array
        transformed_image_np = transformed_image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        transformed_image_np = (transformed_image_np * 255).astype(np.uint8)  # Scale to [0, 255]

        # Save the original and transformed images
        original_save_path = os.path.join(save_path_folder, img)
        transformed_save_path = os.path.join(save_path_folder, f"transformed_{img}")

        cv2.imwrite(original_save_path, image)
        cv2.imwrite(transformed_save_path, transformed_image_np)

        # print(f"Transformed image saved to {transformed_save_path}")


def evaluate_saved_model(model_path, transform, base_model):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check device
    model = classifier(model_path, base_model)
    model.to(device)

    num_classes = 2  # specify num_classes over here


    # model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    train_dataloader, test_dataloader = data_set_up(transform)

    with torch.no_grad():  # Deactivate autograd for evaluation
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Generate classification report
    class_report = classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(num_classes)],
                                         zero_division=0)

    # Extract precision, recall, and f1 for macro average
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"accuracy: {accuracy}, precision: {precision} , {recall}, f1: {f1} ")
    print(class_report)
    return accuracy, precision, recall, f1, class_report


def show_image(img):
    display(img)


def show_random_image(dir):
    subdirs = ['truck_1', 'truck_2']
    random_subdir = random.choice(subdirs)

    files = os.listdir(os.path.join(dir, random_subdir))
    random_index = random.randint(0, len(files) - 1)

    return os.path.join(dir, random_subdir, files[random_index])


def not_truck_detect(dir):
    out_truck_dir = '../../datasets/test_data'
    truck_1 = os.path.join(dir, 'truck_1')
    truck_2 = os.path.join(dir, 'truck_2')

    get_pic1 = os.listdir(truck_1)
    get_pic2 = os.listdir(truck_2)

    for pic1 in get_pic1:
        if pic1.split('_')[1] != 'truck':
            print(pic1)

    for pic2 in get_pic2:
        if pic2.split('_')[1] != 'truck':
            print(pic2)


# scr_path = '/home/user/NgoQuang/datasets/test_data_original'
# des_path = '/home/user/NgoQuang/classification_truck_datasets'


def copy_file(src, des):
    subdirs = ['truck_1', 'truck_2']

    for subdir in subdirs:
        src_subdir_path = os.path.join(src, subdir)
        des_subdir_path = os.path.join(des, subdir)

        all_files = os.listdir(src_subdir_path)

        for file in all_files:
            src_path = os.path.join(src_subdir_path, file)
            dst_path = os.path.join(des_subdir_path, file)
            shutil.copy2(src_path, dst_path)


def check_how_many_pics(img_dir):
    print(f'This directory contains {len(os.listdir(img_dir))} images')
    return len(os.listdir(img_dir))


def move(img_dir):
    os.makedirs("/home/user/NgoQuang/datasets/processed_pics/truck_1", exist_ok=True)
    os.makedirs("/home/user/NgoQuang/datasets/processed_pics/truck_2", exist_ok=True)
    dir_move = '/home/user/NgoQuang/classification_truck_datasets'
    processed_pics = '/home/user/NgoQuang/datasets/processed_pics/'
    subdirs = ['truck_1', 'truck_2']

    name_pics = img_dir.split('/')[-1]
    current_subdir = img_dir.split('/')[-2]
    move_subdir = [subdir for subdir in subdirs if subdir != current_subdir]

    # print(os.path.join(dir_move, current_subdir, name_pics), os.path.join(dir_move, move_subdir[0], name_pics))
    print(f'moved file from {os.path.join(dir_move, current_subdir, name_pics)} to '
          f'{os.path.join(dir_move, move_subdir[0], name_pics)}')

    shutil.copy2(os.path.join(dir_move, current_subdir, name_pics),
                 os.path.join(dir_move, move_subdir[0], name_pics))

    print(f'moved file from {os.path.join(dir_move, current_subdir, name_pics)} to '
          f'{os.path.join(processed_pics, current_subdir, name_pics)}')

    shutil.copy2(os.path.join(dir_move, current_subdir, name_pics),
                 os.path.join(processed_pics, current_subdir, name_pics))

    print(f'deleted image {name_pics} from {os.path.join(dir_move, current_subdir)} ')
    os.remove(os.path.join(dir_move, current_subdir, name_pics))

    dataset = '/home/user/NgoQuang/datasets'
    dataset_subdirs = os.listdir(dataset)
    for dataset_subdir in dataset_subdirs:
        if dataset_subdir.split('_')[0] == 'threshold':
            if name_pics in os.listdir(os.path.join(dataset, dataset_subdir, current_subdir)):
                print(f"Deleted {name_pics} in {os.path.join(dataset, dataset_subdir, current_subdir)}")
                os.remove(os.path.join(dataset, dataset_subdir, current_subdir, name_pics))


def delete(img_dir):
    dir_del = '/home/user/NgoQuang/classification_truck_datasets'
    processed_pics = '/home/user/NgoQuang/datasets/processed_pics/'
    name_pics = img_dir.split('/')[-1]
    current_subdir = img_dir.split('/')[-2]

    print(f'moved file from {os.path.join(dir_del, current_subdir, name_pics)} to '
          f'{os.path.join(processed_pics, current_subdir, name_pics)}')

    shutil.copy2(os.path.join(dir_del, current_subdir, name_pics),
                 os.path.join(processed_pics, current_subdir, name_pics))

    print(f"Deleted {name_pics} in {os.path.join(dir_del, current_subdir)}")
    os.remove(os.path.join(dir_del, current_subdir, name_pics))

    dataset = '/home/user/NgoQuang/datasets'
    dataset_subdirs = os.listdir(dataset)
    for dataset_subdir in dataset_subdirs:
        if dataset_subdir.split('_')[0] == 'threshold':
            if name_pics in os.listdir(os.path.join(dataset, dataset_subdir, current_subdir)):
                print(f"Deleted {name_pics} in {os.path.join(dataset, dataset_subdir, current_subdir)}")
                os.remove(os.path.join(dataset, dataset_subdir, current_subdir, name_pics))


def delete_img_first_img(img_dir):
    for img in os.listdir(img_dir):
        if img.startswith('img'):
            img_path = os.path.join(img_dir, img)
            os.remove(img_path)


def split_train_test_dataset(src, img_dir, ratio):

    for split in ['train', 'test']:
        for truck in ['truck_1', 'truck_2']:
            os.makedirs(os.path.join(img_dir, split, truck), exist_ok=True)

    test_dir = os.path.join(img_dir, 'test')
    train_dir = os.path.join(img_dir, 'train')

    src_subdirs = ['truck_1', 'truck_2']

    for subdir in src_subdirs:
        path_src_subdir = os.path.join(src, subdir)
        num_train = int(ratio * len(os.listdir(path_src_subdir)))

        # Move file for each truck files

        for img in os.listdir(path_src_subdir)[:num_train]:
            shutil.copy2(os.path.join(path_src_subdir, img),
                         os.path.join(train_dir, subdir, img))
            # print(f'Move from {os.path.join(path_src_subdir, img)} to {os.path.join(train_dir, subdir, img)}')
        print(f'Move successfully from {path_src_subdir} to {os.path.join(train_dir, subdir)}')

        for img in os.listdir(path_src_subdir)[num_train:]:

            shutil.copy2(os.path.join(path_src_subdir, img),
                         os.path.join(test_dir, subdir, img))
            # print(f'Move from {os.path.join(path_src_subdir, img)} to {os.path.join(test_dir, subdir, img)}')
        print(f'Move successfully from {path_src_subdir} to {os.path.join(test_dir, subdir)}')


def delete_file(file):
    if os.path.isfile(file):
        bin = os.path.join('/home/user/Quang/datasets', 'bin')
        os.makedirs(bin, exist_ok=True)
        shutil.move(file, bin)
        # os.remove(file)
        print("Delete successfully")
    else:
        print("File is not exists")


def delete_models(list_model):
    path_models = '/home/user/Quang/truck_classification/models'
    for model_order in list_model:
        model = 'model' + model_order + '.pth'
        path_model = os.path.join(path_models, model)

        if model in os.listdir(path_models):
            delete_file(path_model)
            print(f'delete successfully {path_model}')
        else:
            print(f'{path_model} not in this path')
            continue

def delete_folder(folder):
    if os.path.isdir(folder):
        bin = os.path.join('/home/user/Quang/datasets', 'bin')
        os.makedirs(bin, exist_ok=True)
        shutil.move(folder, bin)
        print("Delete successfully")
    else:
        print("File is not exists")


def clean_bin():
    bin = os.path.join('/home/user/Quang/datasets', 'bin')
    for file in os.listdir(bin):
        os.remove(os.path.join(bin, file))
    print("Clean bin successfully")


def step_back_delete(file, destinate):
    dir_bin = '/home/user/Quang/datasets/bin'
    if file in os.listdir(dir_bin):
        path_file = os.path.join(dir_bin, file)
        shutil.move(path_file, destinate)


def clean_memory():
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
