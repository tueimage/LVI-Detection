

import os
import sys
from pathlib import Path

# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))

import torch # type: ignore
import numpy as np
import argparse

from gigapath.classification_head import get_model  # type: ignore
from metrics import calculate_metrics_with_task_cfg
from task_configs.utils import load_task_config
from utils import seed_torch, get_splits, get_loader
from datasets.slide_datatset import SlideDataset
from utils import get_loss_function, get_records_array
import pandas as pd # type: ignore

# Evaluation settings
parser = argparse.ArgumentParser(description='Configurations for WSI Evaluation')
parser.add_argument('--root_path', type=str, default=None,  help='data directory')
parser.add_argument('--save_exp_code', type=str, default=None,  help='experiment code to save eval results')
parser.add_argument('--seed', type=int, default=2021,   help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--models_exp_code', type=str, default=None,    help='experiment code to load trained models')
parser.add_argument('--split_dir', type=str, default=None,  help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--fp16',           action='store_true', default=True, help='Fp16 training')
# model settings
parser.add_argument('--model_arch',     type=str, default='longnet_enc12l768d')
parser.add_argument('--input_dim',      type=int, default=1536, help='Dimension of input tile embeddings')
parser.add_argument('--latent_dim',     type=int, default=768, help='Hidden dimension of the slide encoder')
parser.add_argument('--feat_layer',     type=str, default='12', help='The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers')
parser.add_argument('--pretrained',     type=str, default='', help='Pretrained GigaPath slide encoder')
parser.add_argument('--freeze',         action='store_true', default=False, help='Freeze pretrained model')
parser.add_argument('--global_pool',    action='store_true', default=False, help='Use global pooling, will use [CLS] token if False')
parser.add_argument('--dropout',       type=float, default=0.1, help='Dropout rate')

args = parser.parse_args()

args.dataset_csv = '/home/20215294/Implementations/gigapath/dataset_csv/LVI/lvi.csv'
args.root_path = "/home/20215294/Data/LVI/patches_20x/feat/h5_files/"
args.split_dir = "/home/20215294/Implementations/gigapath/dataset_csv/LVI/"
args.task_cfg_path = "./task_configs/lvi.yaml"
args.models_exp_code = "/home/20215294/Implementations/gigapath/outputs/LVI/lvi_finetune/lvi/run_blr-_wd-0.01_ld-0.95_feat-12_dropout-0.01_not_freeze_aug_fixed/eval_pretrained_lvi/"
args.save_exp_code = "not_freeze_aug_fixed"
args.pretrained = "/home/20215294/.cache/slide_encoder.pth"
args.model_arch = "gigapath_slide_enc12l768d"
args.input_dim=1536
args.latent_dim=768
args.max_wsi_size = 198656
args.tile_size = 2048
args.freeze = False
args.dropout = 0.01

def evaluate(loader, model, fp16_scaler, loss_fn, args):
    model.eval()

    # set the evaluation records
    records = get_records_array()
    # get the task setting
    task_setting = args.task_config.get('setting', 'multi_label')

    all_probs= {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # load the batch and transform this batch
            images, img_coords, label = batch['imgs'], batch['coords'], batch['labels']
            images = images.to(args.device, non_blocking=True)
            img_coords = img_coords.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True).long()
            slide_id =  batch['slide_id']
            with torch.cuda.amp.autocast(fp16_scaler is not None, dtype=torch.float16):
                # get the logits
                logits, _ = model(images, img_coords)

                # get the loss
                if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                    label = label.squeeze(-1).float()
                else:
                    label = label.squeeze(-1).long()

                # reshape the label
                label = label.reshape(-1, 1)
                loss = loss_fn(logits, label)
                loss = torch.mean(loss)

            # update the records
            records['loss'] += loss.item()
            if task_setting == 'multi_label':
                Y_prob = torch.sigmoid(logits)
                records['prob'].append(Y_prob.detach().cpu().numpy())
                records['label'].append(label.detach().cpu().numpy())

            all_probs.update({slide_id[0]: [img_coords.detach().cpu().numpy(), label.detach().cpu().numpy(), Y_prob.detach().cpu().numpy()]})
            print(f"{slide_id[0]} is done!")

    records.update(calculate_metrics_with_task_cfg(records['prob'], records['label'], args.task_config))
    records['loss'] = records['loss'] / len(loader)

    if task_setting == 'multi_label':
        info = 'Loss: {:.4f}, Micro FROC: {:.4f},  Micro AUPRC: {:.4f}'.format(records['loss'], records['micro_froc'], records['micro_auprc'])
    print(info)

    return records, all_probs




if __name__ == '__main__':
    print("started!")

    print(args)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print('Using fp16 evaluation')


    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # set the loss function
    loss_fn = get_loss_function(args.task_config)

    # set the split key
    args.split_key = 'slide_id'

    # set up the dataset
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file

    # use the slide dataset
    DatasetClass = SlideDataset

    # set up the results dictionary
    results = {}

    # set up the fold directory
    fold= 0
    save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
    model_dir = os.path.join(args.models_exp_code, 'fold_' + str(fold), "checkpoint.pt")

    # create the save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # get the splits
    train_splits, val_splits, test_splits = get_splits(dataset, fold=0, **vars(args))

    # instantiate the dataset
    train_data, val_data, test_data = DatasetClass(dataset, args.root_path, train_splits, args.task_config, split_key=args.split_key, train= True) \
                                    , DatasetClass(dataset, args.root_path, val_splits, args.task_config, split_key=args.split_key) if len(val_splits) > 0 else None \
                                    , DatasetClass(dataset, args.root_path, test_splits, args.task_config, split_key=args.split_key) if len(test_splits) > 0 else None

    # get the dataloader
    train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))

    # set up the model
    model = get_model(**vars(args))
    model = model.to(args.device)

    # set up the loss function
    loss_fn = get_loss_function(args.task_config)
    
    print('Training on {} samples'.format(len(train_loader.dataset)))
    print('Validating on {} samples'.format(len(val_loader.dataset))) if val_loader is not None else None
    print('Testing on {} samples'.format(len(test_loader.dataset))) if test_loader is not None else None
    print('Evaluation starts!')

    # start evaluating
    # load model for test
    model.load_state_dict(torch.load(model_dir))

    # validate the model
    val_records, all_probs_val = evaluate(val_loader, model, fp16_scaler, loss_fn, args)

    # test the model
    test_records, all_probs_test = evaluate(test_loader, model, fp16_scaler, loss_fn, args)

    # update the results
    records = {'test': test_records}
    for record_ in records:
        for key in records[record_]:
            if 'prob' in key or 'label' in key:
                continue
            key_ = record_ + '_' + key
            if key_ not in results:
                results[key_] = []
            results[key_].append(records[record_][key])

    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)

    print('\n')
    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))

    print('\nResults saved in: {}'.format(os.path.join(save_dir, 'summary.csv')))
        
    # save the all_probs into seperate npy files
    np.save(os.path.join(save_dir, 'all_probs_val.npy'), all_probs_val)
    np.save(os.path.join(save_dir, 'all_probs_test.npy'), all_probs_test)
    print("Done!")