from modular import utils
from choice_commands import efficientNetWeight_choice, model_choice, transform_type_choice

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
