
import os
import importlib


from backbones import resnet
from backbones.nn import MLP, Shallow


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]
names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, encoder, n_classes):
    return names[args.model](encoder, n_classes=n_classes, args=args) # only discriminative

def get_teacher_model(args, encoder, n_classes):
    return names[args.teacher](encoder, n_classes=n_classes, args=args) # only discriminative


def get_teacher_encoder(args):
    match args.teacher_backbone:
        case 'mlp':
            backbone = MLP(input_dim=2)
        case 'resnet18':
            backbone = resnet.ResNet18(final_dim=args.final_dim)
        case 'resnet34':
            backbone = resnet.ResNet34(final_dim=args.final_dim)
        case 'resnet50':
            backbone = resnet.ResNet50(final_dim=args.final_dim)
        case 'shallow':
            backbone = Shallow(input_dim=768, hidden_units=2048, repr_dim=10)
        case _:
            raise ValueError(f'Chosen backbone not recognized: {args.teacher_backbone}')
    return backbone
