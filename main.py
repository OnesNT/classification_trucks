import argparse
import os

import torch.cuda

from modular import utils
from modular import train
from modular import draw_graph
import torchvision

def model_choice(args):

    if args.model_choice is None:
        print("Error: model choice is require")

    if args.model_choice == 'r':
        base_model = train.base_modelResNet34
        version_model = 34
        print("Using base_modelResNet34")
    elif args.model_choice == 'c':
        base_model = train.base_modelWeightConvNeXt
        version_model = 'base'
        print("Using base_modelWeightConvNeXt")
    elif args.model_choice == 'e':
        base_model, version_model = efficientNetWeight_choice(args)
    elif args.model_choice == 'e_v2':
        base_model = train.base_v2_modelEfficientNet_S
        version_model = 'S'
    else:
        print("Invalid transform choice")
        return

    return args.model_choice, base_model, version_model


def efficientNetWeight_choice(args):
    # get version model
    if args.version_model is None:
        print("Error: --EfficientNet version is require")

    if args.version_model == 0:
        base_model = train.base_model0
        print("Using EfficientNetB0")
    elif args.version_model == 1:
        base_model = train.base_model1
        print("Using EfficientNetB1")
    elif args.version_model == 2:
        base_model = train.base_model2
        print("Using EfficientNetB2")
    # elif args.version_model == 3:
    #     base_model = train.base_v2_modelEfficientNet_M
    else:
        print("Invalid transform choice")
        return None, None
    return base_model, args.version_model


def transform_type_choice(args):
    if args.transform_choice is None:
        print("Error: --transform-choice is required")
        return
    # Set transform based on user choice
    if args.transform_choice == 1:
        transform = train.transform1
        print("Using transform1")
    elif args.transform_choice == 2:
        transform = train.transform2
        print("Using transform2")
    elif args.transform_choice == 3:
        transform = train.transform3
        print("Using transform3")
    else:
        print("Invalid transform choice")
        return
    return transform

def schedule_lr_choice(args):
    if args.schedule_lr is None:
        print('Error: --schedule-lr is required')
        return

    if args.schedule_lr not in [0, 1]:
        print("Error: invalid choice for --schedule-lr")
        return

    return args.schedule_lr



def utils_commands(args):
    # python main.py --check-how-many-img
    if args.check_how_many_img:
        utils.check_how_many_pics(args.check_how_many_img)

    # python main.py --split_test_train
    if args.split_test_train:
        src = '/home/user/Quang/classification_truck_datasets'
        img_dir = src
        ratio = 0.8
        utils.split_train_test_dataset(src, img_dir, ratio)

    # python main.py --delete-img-image
    if args.delete_img_image:
        truck_1 = '/home/user/Quang/classification_truck_datasets/truck_1'
        truck_2 = '/home/user/Quang/classification_truck_datasets/truck_2'

        print('deleting truck 1 ')
        utils.delete_img_first_img(truck_1)
        print('deleting truck 2 ')
        utils.delete_img_first_img(truck_2)
        print('finish deleting')

    # Conditional execution based on the flag
    # python main.py --show-image '/home/user/NgoQuang/datasets/threshold_below_0.6/truck_1/...'
    if args.show_image:
        print(f"Showing a random image from the directory: {args.show_image}")
        print(utils.show_random_image(args.show_image))

    #  python main.py --delete-image '/home/user/NgoQuang/datasets/threshold_below_0.6/truck_1/...'
    if args.delete_image:
        print(f"Deleted image {args.delete_image.split('/')[-1]}")
        utils.delete(args.delete_image)

    #  python main.py --move-image '/home/user/NgoQuang/datasets/threshold_below_0.6/truck_1/...'
    if args.move_image:
        # print(args.move_image)
        print(f"Moved image {args.move_image.split('/')[-1]}")
        utils.move(args.move_image)

    # python main.py --evaluate-model '/home/user/NgoQuang/truck_classification/modular/models/model7.pth' --version-model 1 --transform-choice 1
    if args.evaluate_model:
        transform = transform_type_choice(args)
        base_model = efficientNetWeight_choice(args)
        print(f"Evaluate model from {args.evaluate_model}")
        utils.evaluate_saved_model(args.evaluate_model, transform, base_model)

    # python main.py --get-content '/home/user/NgoQuang/truck_classification/modular/models/model7.pth' --version-model 1 --transform-choice 1
    if args.get_content:
        model, base_model, version_model = model_choice(args)
        print(f"get content model from {args.get_content}")
        model_path = args.get_content
        schedule_lr = args.schedule_lr
        utils.load_model(model_path, model, base_model, schedule_lr)

    # python main.py --delete-file '/home/user/NgoQuang/truck_classification/modular/models/model7.pth'
    if args.delete_file:
        dir = args.delete_file
        utils.delete_file(dir)

    # python main.py --delete-models 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    if args.delete_models:
        dir = args.delete_models
        list_model = dir.split(',')
        # print(list_model)
        # print(dir)

        utils.delete_models(list_model)

    if args.delete_folder:
        dir = args.delete_file
        utils.delete_folder(dir)

    # python main.py --clean-bin
    if args.clean_bin:
        utils.clean_bin()

    if args.step_back_delete:
        file = args.step_back_detele
        destination = args.destination
        utils.step_back_delete(file, destination)

    if args.clean_memory:
        utils.clean_memory()

    return 0


def train_commands(args):

    # python main.py --inference-truck-1 DIR_MODEL --version-model 1
    if args.inference_truck_1:
        truck_dir = '/home/user/Quang/datasets/01-07_08_24_all_cameras/truck_1'
        out_truck_dir = '/home/user/Quang/datasets/inference_check'
        model_path = args.inference_truck_1
        schedule_lr = args.schedule_lr
        model_type, base_model, version_model = model_choice(args)

        train.inference_for_truck_1(truck_dir, out_truck_dir, model_path, base_model, model_type, schedule_lr)
    # python main.py --detect-and-train --version-model 1
    if args.detect_and_train:
        truck_dir = '/home/user/Quang/datasets/truck'
        out_truck_dir = '/home/user/Quang/datasets/test_data_original'
        model_path = '/home/user/Quang/truck_classification/modular/models/model1.pth'
        detector_path = '/home/user/Quang/truck_classification/weights/yolov8m/7/best.pt'

        base_model = efficientNetWeight_choice(args)
        train.detect_and_classify_truck(truck_dir, out_truck_dir, model_path, base_model, detector_path)

    # Conditional execution based on the flag
    #  python main.py --train-model e --transform-choice 1 --version_model 1
    #  python main.py --train-model r --transform-choice 1
    #  python main.py --train-model c --transform-choice 1
    if args.train_model:
        print(f"Training model with images from: {args.train_model}")
        # Check if transform_choice is provided when training
        transform = transform_type_choice(args)
        choice_model, base_model, version_model = model_choice(args)
        schedule_lr = schedule_lr_choice(args)
        train.train_model(transform, choice_model, base_model, version_model, schedule_lr)


def draw_commands(args):
    # python main.py --draw-graph DIR-MODEL --version-model 1
    if args.draw_graph:
        choice_model, model_base, version_model = model_choice(args)
        schedule_lr = args.schedule_lr
        print(f"Drawing graph with model from: {args.draw_graph}")
        draw_graph.load_and_draw(args.draw_graph, model_base, choice_model, schedule_lr)
    return 0


def main():

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Truck classification bash command")

    # Add a flag for showing a random image, with a required directory argument

    # Argument for Utils command
    parser.add_argument('--delete-models',
                        metavar='model_number',
                        type=str,
                        help='delete model by numbers')
    parser.add_argument('--destination',
                        metavar='DIR',
                        type=str,
                        help='destination of moving file')
    parser.add_argument('--step-back-delete',
                        metavar='DIR',
                        type=str,
                        help='step back deleting a file')
    parser.add_argument('--get-content',
                        metavar='DIR',
                        type=str,
                        help='get content from saved model')
    parser.add_argument('--split-test-train',
                        action='store_true',
                        help='split file to train and test')
    parser.add_argument('--clean-memory',
                        action='store_true',
                        help='clean gpu')
    parser.add_argument('--check-how-many-img',
                        metavar='DIR',
                        type=str,
                        help='check how many images in a directory')
    parser.add_argument('--delete-img-image',
                        action='store_true',
                        help='delete image with img at beginning')

    parser.add_argument('--show-image',
                        metavar='DIR',
                        type=str,
                        help='Show a random image from the specified directory')

    parser.add_argument('--delete-image',
                        metavar='DIR',
                        type=str,
                        help='Delete images making distort data')

    parser.add_argument('--move-image',
                        metavar='DIR',
                        type=str,
                        help='move truck_1 to truck_2 or converse')

    parser.add_argument('--evaluate-model',
                        metavar='DIR',
                        type=str,
                        help='evaluate model')

    parser.add_argument('--delete-file',
                        metavar='DIR',
                        type=str,
                        help='delete a file')

    parser.add_argument('--delete-folder',
                        metavar='DIR',
                        type=str,
                        help='delete a folder'
                        )

    parser.add_argument('--clean-bin',
                        action='store_true',
                        help='clean bin')

    # par
    # Argument for Training commands
    parser.add_argument('--schedule-lr',
                        type=int,
                        choices=[0, 1],
                        help='Use schedule lr technique or not')

    parser.add_argument('--inference-truck-1',
                        metavar='DIR',
                        type=str,
                        help='check inference with truck 1 file')

    parser.add_argument('--model-choice',
                        metavar='DIR',
                        type=str,
                        help='choose model')

    parser.add_argument('--detect-and-train',
                        action='store_true',
                        help='Directory containing input images')

    parser.add_argument('--train-model',
                        action='store_true',
                        help='Directory containing input images')

    # Argument for Drawing graph
    parser.add_argument('--draw-graph',
                        metavar='DIR',
                        type=str,
                        help='Draw graph from saved model')

    # Argument for choice
    parser.add_argument('--output-dir', metavar='DIR', help='Directory to save output images')
    parser.add_argument('--input-dir')

    # Add argument for transform choice
    parser.add_argument('--transform-choice', type=int, choices=[1, 2, 3],
                        help='Choose the transform to apply: 1 for transform1, 2 for transform2')

    # Add argument for efficient net version choice
    parser.add_argument('--version-model', type=int, choices=[0, 1, 2, 3],
                        help='Choose the transform to apply: 0 for efficientNetB0, 1 for efficientNetB1, 2 for efficientNetB2')

    # Parse the arguments
    args = parser.parse_args()

    # Call utility functions
    utils_commands(args)

    # Call training functions
    train_commands(args)

    # Call drawing function
    draw_commands(args)


if __name__ == '__main__':
    main()

    # print(torchvision.models.list_models())
