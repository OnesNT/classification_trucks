from modular import train
from commands.choice_commands import efficientNetWeight_choice, model_choice, transform_type_choice, schedule_lr_choice


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
        choice = args.optimizer_choice
        schedule_lr = schedule_lr_choice(args)
        train.train_model(transform, choice_model, base_model, version_model, schedule_lr, choice)

