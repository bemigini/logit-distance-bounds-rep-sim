from argparse import ArgumentParser
from dsets import NAMES as DATASET_NAMES
from models import get_all_models



def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    # dataset
    parser.add_argument('--dataset', default='synthetic', type=str, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--min_radius', type=float, default=5, help='Minimum radius of the circles for SYNTHETIC.')
    parser.add_argument('--max_radius', type=float, default=5, help='Maximum radius of the circles for SYNTHETIC.')
    parser.add_argument('--modality', default='standard', type=str, choices=['standard', 'interpolation', 'extrapolation'],
                    help='Whether to test interpolation or extrapolation for SYNTHETIC dataset.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers for dataloaders.')
    parser.add_argument('--data_dir', type=str, default='data', help='The directory to save/load data from.')
    
    
    # model settings
    parser.add_argument('--model', type=str, default="classifier", help='Model name.', choices=get_all_models())
    parser.add_argument('--teacher', type=str, default="classifier", help='Model name.', choices=get_all_models())
    parser.add_argument('--entropy', action='store_true', default=False, help='Activate entropy on batch.')
    parser.add_argument('--backbone', type=str, default="", help='Choose backbone for model to train.')
    parser.add_argument('--teacher_backbone', type=str, default="", help='Choose backbone for the teacher model.')
    parser.add_argument('--loss_type', type=str, default="l1", help='loss type, l1 or l2.', choices=['l1', 'l2'])
    # additional hyperparams
    parser.add_argument('--w_rec', type=float, default=1, help='Weight of Reconstruction')
    parser.add_argument('--beta',  type=float, default=2, help='Multiplier of KL')
    parser.add_argument('--w_h',   type=float, default=1, help='Weight of entropy')
    parser.add_argument('--w_c',   type=float, default=1, help='Weight of concept sup')
    parser.add_argument('--final_dim', type=int, default=2, help='Dimension of the embeddings and unembeddings.')
    
    # optimization params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Warmup epochs.')
    parser.add_argument('--exp_decay', type=float, default=0.99, help='Exp decay of learning rate.')
    
    # learning hyperams
    parser.add_argument('--n_epochs',   type=int, default=50, help='Number of epochs per task.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    # random seed
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    # verbosity
    parser.add_argument('--notes', type=str, default=None, help='Notes for this run.')
    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--generate_gif', action='store_true', default=False, help='Generate gif of data distribution over time.')
    # logging
    parser.add_argument('--all_metrics',  action='store_true', default=False,  help='get all metrics every 5 or 20 epochs depending on num epochs')
    # checkpoints
    parser.add_argument('--checkin',    type=str, default=None, help='location and path FROM where to load ckpt.' )    
    parser.add_argument('--checkout',   type=str, default=None, help='location and path  TO  where to store ckpt.' )  
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='Path to load teacher model from.')    
    # post-hoc evaluation
    parser.add_argument('--posthoc',  action='store_true', default=False, help='Used to evaluate only the loaded model')
    parser.add_argument('--validate', action='store_true', default=False, help='Used to evaluate on the validation set for hyperparameters search')
