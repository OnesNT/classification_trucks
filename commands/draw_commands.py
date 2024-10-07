from modular import draw_graph
from choice_commands import model_choice

def draw_commands(args):
    # python main.py --draw-graph DIR-MODEL --version-model 1
    if args.draw_graph:
        choice_model, model_base, version_model = model_choice(args)
        schedule_lr = args.schedule_lr
        print(f"Drawing graph with model from: {args.draw_graph}")
        draw_graph.load_and_draw(args.draw_graph, model_base, choice_model, schedule_lr)
    return 0