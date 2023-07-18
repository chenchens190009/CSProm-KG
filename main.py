import os
import argparse
from datetime import datetime
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
from transformers import AutoConfig, BertTokenizer
from models.P_model import KGCPromptTuner
from kgc_data import KGCDataModule
from helper import get_num, read, read_name, read_file, get_gt
from callbacks import PrintingCallback


def main():
    if configs.save_dir == '':
        if configs.jobid == '':
            configs.save_dir = os.path.join('./checkpoint', datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        else:
            configs.save_dir = os.path.join('./checkpoint', configs.jobid)
    os.makedirs(configs.save_dir, exist_ok=True)
    trainer_params = {
        'max_epochs': configs.epochs,  # 1000
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,  # 2
        'check_val_every_n_epoch': configs.check_val_every_n_epoch,
        'enable_progress_bar': True,
    }
    if torch.cuda.is_available():
        trainer_params['devices'] = [int(configs.gpu)]
        trainer_params['accelerator'] = 'gpu'
    else:
        trainer_params['accelerator'] = 'cpu'

    if configs.accumulate_grad_batches > 1:
        trainer_params['accumulate_grad_batches'] = configs.accumulate_grad_batches
    if configs.use_fp16:
        trainer_params['precision'] = 16

    ## read triples
    train = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')
    test = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train + valid + test

    ## construct ground truth dictionary
    # ground truth .shape: dict, example: {hr_str_key1: [t_id11, t_id12, ...], (hr_str_key2: [t_id21, t_id22, ...], ...}
    train_tail_gt, train_head_gt = get_gt(configs, train)
    all_tail_gt, all_head_gt = get_gt(configs, all_triples)

    gt = {
        'train_tail_gt': train_tail_gt,
        'train_head_gt': train_head_gt,
        'all_tail_gt': all_tail_gt,
        'all_head_gt': all_head_gt,
    }

    ent_names, rel_names = read_name(configs, configs.dataset_path, configs.dataset)
    ent_descs = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt', 'desc')
    tok = BertTokenizer.from_pretrained(configs.pretrained_model, add_prefix_space=False)

    text_dict = {
        'ent_names': ent_names,
        'rel_names': rel_names,
        'ent_descs': ent_descs,
    }
    ## construct datamodule
    datamodule = KGCDataModule(configs, train, valid, test, text_dict, tok, gt)
    print('datamodule construction done.', flush=True)

    ## construct trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mrr',
        dirpath=configs.save_dir,
        filename=configs.dataset + '-{epoch:03d}-{' + "val_mrr" + ':.4f}',
        mode='max'
    )
    printing_callback = PrintingCallback()
    trainer_params['callbacks'] = [
        checkpoint_callback,
        printing_callback,
    ]
    trainer = pl.Trainer(**trainer_params)

    ## construct model parameters
    kw_args = {
        'text_dict': text_dict,
        'gt': gt,
    }

    if configs.model_path == '':
        if configs.continue_path == '':
            model = KGCPromptTuner(configs, **kw_args)
        else:
            model = KGCPromptTuner.load_from_checkpoint(configs.continue_path, strict=False, configs=configs, **kw_args)
        print('model construction done.', flush=True)
        trainable_params, non_trainable_params = 0, 0
        for name, params in model.named_parameters():
            if params.requires_grad:
                print('name:', name, 'shape:', params.shape, 'numel:', params.numel())
                trainable_params += params.numel()
            else:
                non_trainable_params += params.numel()
        print('trainable params:', trainable_params, 'non trainable params:', non_trainable_params)
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
    else:
        model_path = configs.model_path
    print('model_path:', model_path, flush=True)
    model = KGCPromptTuner.load_from_checkpoint(model_path, strict=False, configs=configs, **kw_args)
    trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    transformers.logging.set_verbosity_error()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./data/processed')
    parser.add_argument('-jobid', type=str, default='XXXXXXXX')
    parser.add_argument('-dataset', dest='dataset', default='WN18RR',
                        help='Dataset to use, default: InferWiki16k')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='t5-base', help='')
    parser.add_argument('-batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')

    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    parser.add_argument('-desc_max_length', default=0, type=int, help='')
    parser.add_argument('-prompt_length', default=0, type=int, help='')
    parser.add_argument('-embed_dim', default=0, type=int, help='')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=8, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=16, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-label_smoothing', default=0., type=float, help='Label smoothing')
    parser.add_argument('-use_log_ranks', action='store_true', help='')
    parser.add_argument('-n_lar', default=0, type=int, help='Number of LARs')
    parser.add_argument('-gamma', default=1., type=float, help='Margin loss: margin between pos and neg')
    parser.add_argument('-alpha', default=0., type=float, help='Weight between CE loss and margin loss')
    parser.add_argument('-graph_model', default='conve', type=str, help='[null | transe | distmult | conve | rotate]')
    parser.add_argument('-loss_gamma', default=0., type=float, help='Gamma for score function of transe and rotate')
    parser.add_argument('-lr_scheduler', default='linear', type=str, help='[linear | cosine]')
    parser.add_argument('-max_lar_samples', default=-1, type=int, help='Maximum of negative samples')
    parser.add_argument('-continue_path', dest='continue_path', default='', help='The path for continuing training')
    parser.add_argument('-accumulate_grad_batches', default=1, type=int, help='')
    parser.add_argument('-check_val_every_n_epoch', default=3, type=int, help='')
    parser.add_argument('-use_speedup', action='store_true', help='')
    parser.add_argument('-text_len', default=72, type=int, help='')
    parser.add_argument('-use_fp16', action='store_true', help='')
    parser.add_argument('-prompt_hidden_dim', default=-1, type=int, help='')
    parser.add_argument('-alpha_step', default=0., type=float, help='')

    configs = parser.parse_args()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent
    configs.n_rel = n_rel
    configs.vocab_size = AutoConfig.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = AutoConfig.from_pretrained(configs.pretrained_model).hidden_size
    configs.is_temporal = 'ICEWS' in configs.dataset
    if configs.prompt_hidden_dim == -1:
        configs.prompt_hidden_dim = configs.embed_dim // 2
    print(configs, flush=True)

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()
