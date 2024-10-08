# from modular import train
from config import transforms, base_model

def model_choice(args):

    if args.model_choice is None:
        print("Error: model choice is require")

    if args.model_choice == 'r34':
        base_model_choice = base_model.base_modelResNet34
        version_model = 34
        print("Using base_modelResNet34")
    elif args.model_choice == 'r50':
        base_model_choice = base_model.base_modelResNet50
        version_model = 50
        print("Using base_modelResNet50")
    elif args.model_choice == 'c':
        base_model_choice = base_model.tiny_modelWeightConvNeXt
        version_model = 'tiny'
        print("Using base_modelWeightConvNeXt")
    elif args.model_choice == 'e':
        base_model_choice, version_model = efficientNetWeight_choice(args)
    elif args.model_choice == 'e_v2':
        base_model_choice = base_model.base_v2_modelEfficientNet_S
        version_model = 's'
    else:
        print("Invalid model choice")
        return

    return args.model_choice, base_model_choice, version_model


def efficientNetWeight_choice(args):
    # get version model
    if args.version_model is None:
        print("Error: --EfficientNet version is require")

    if args.version_model == 0:
        base_model_choice = base_model.base_model0
        print("Using EfficientNetB0")
    elif args.version_model == 1:
        base_model_choice = base_model.base_model1
        print("Using EfficientNetB1")
    elif args.version_model == 2:
        base_model_choice = base_model.base_model2
        print("Using EfficientNetB2")
    # elif args.version_model == 3:
    #     base_model = train.base_v2_modelEfficientNet_M
    else:
        print("Invalid transform choice")
        return None, None
    return base_model_choice, args.version_model


def transform_type_choice(args):
    if args.transform_choice is None:
        print("Error: --transform-choice is required")
        return
    # Set transform based on user choice
    if args.transform_choice == 1:
        transform = transforms.transform1
        print("Using transform1")
    elif args.transform_choice == 2:
        transform = transforms.transform2
        print("Using transform2")
    elif args.transform_choice == 3:
        transform = transforms.transform3
        print("Using transform3")
    elif args.transform_choice == 4:
        transform = transforms.transform_efficientNetB2
        print("Using transform_efficientNetB2")
    elif args.transform_choice == 5:
        transform = transforms.transform_efficientNet_V2_S
        print("Using transform_efficientNet_V2_S")
    elif args.transform_choice == 6:
        transform = transforms.transform_resnet34
        print("Using transform_resnet34")
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