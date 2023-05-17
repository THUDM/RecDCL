import argparse
import wandb
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    wandb.login(key="")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    wandb.init(
        name='exp-' + args.dataset + '-' + args.model,
        project="SSL4Rec")
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
