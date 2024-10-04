import argparse

def parsers():
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

    return parser