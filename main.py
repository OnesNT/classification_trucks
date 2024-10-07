from commands import parser_commands
from commands.utils_commands import utils_commands
from commands.train_commands import train_commands 
from commands.draw_commands import draw_commands


def main():

    parser = parser_commands.parsers()

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
