import os
import yaml
import argparse

from torch.optim.optimizer import Optimizer
import wandb
import numpy as np
from importlib import import_module

from utils import CutMix_half, empty_cache, seed_everything, get_lr, increment_path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from loss import create_criterion

from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

def train(config_train):
    # gpu를 비우고 seed를 고정
    empty_cache()
    seed_everything(config_train['seed'])

    # model을 저장할 path를 지정
    save_dir = increment_path(os.path.join(config_train['model_dir'], config_train['name']))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if config_train['wandb']:
        wandb.init(project=config_train['wandb_proj'], entity=config_train['wandb_entity'])

    # -- dataset
    dataset_module = getattr(import_module("dataset"), config_train['dataset'])  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=config_train['data_dir'], # images directory
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), config_train['augmentation'])  # default: BaseAugmentation
    
    transform  = transform_module(
                            mean=dataset.mean,
                            std=dataset.std,
                            is_valid=False,
                            age_labels=dataset.age_labels
                        )

    transform_valid = transform_module(
        mean=dataset.mean,
        std=dataset.std,
        is_valid=True,
        age_labels=dataset.age_labels
    )

    labels = dataset.all_labels

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=config_train['batch_size'],
        num_workers=config_train['num_workers'],
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config_train['valid_batch_size'],
        num_workers=config_train['num_workers'],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # --earlystopping
    if config_train['earlystopping']:
        earlystop_module = getattr(import_module("utils"), 'EarlyStopping')
        earlystopping = earlystop_module(patience=config_train['patience'], 
                                         verbose=config_train['verbose'], 
                                         path=save_dir)

    # -- model
    model_module = getattr(import_module("model"), config_train['model'])  # default: BaseModel
    model = model_module(
        num_classes=num_classes,
        freeze = False
    ).to(device)
    
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(config_train['criterion'], classes=num_classes)  # default: f1
    opt_module = getattr(import_module("torch.optim"), config_train['optimizer'])  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config_train['lr'],
        weight_decay=config_train['lr_decay_step']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    best_val_acc = 0
    best_val_loss = np.inf

    for epoch in tqdm(range(config_train['epochs'])):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        train_f1 = 0.
        for idx, train_batch in enumerate(train_loader):
            if config_train['cutmix']:
                inputs, labels, index = train_batch
                inputs = inputs
                inputs, active_cutmixs = transform(inputs, index)
                inputs = inputs.to(device)
                inputs, lam, rand_indexs = CutMix_half(inputs, active_cutmixs)()
                labels = labels.to(device)
                rand_target = torch.tensor([labels[idx] for idx in rand_indexs], device=device)
            else:
                inputs, labels, _ = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            if config_train['cutmix']:
                loss = criterion(outs, labels) * lam + criterion(outs, rand_target) * (1 - lam)
            else:
                loss = criterion(outs, labels)
                
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            train_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            
            if (idx + 1) % config_train['log_interval'] == 0:
                train_loss = loss_value / config_train['log_interval']
                train_acc = matches / config_train['batch_size'] / config_train['log_interval']
                train_f1 = train_f1 / config_train['log_interval'] # 현재 interval 동안의 train_f1 계산
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{config_train['epochs']}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} || "
                    f"training F1 Score {train_f1:4.4}"
                )

                if config_train['wandb']:
                    wandb.log({'train/accuracy': train_acc, 'train/loss': train_loss, 'train/f1_score' : train_f1})

                loss_value = 0
                matches = 0
                train_f1 = 0.

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            # figure = None
            valid_f1, n_iter = 0., 0.
            for val_batch in val_loader:
                inputs, labels, index = val_batch
                inputs, _ = transform_valid(image=inputs, index=index)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                valid_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                n_iter += 1
                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            valid_f1 = valid_f1 / n_iter
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"validation F1 Score {valid_f1:4.4}"
            )
            
            if config_train['wandb']:
                wandb.log({'train/accuracy': train_acc, 'train/loss': train_loss, 'train/f1_score' : train_f1,
                            'vaild/accuracy': val_acc, 'vaild/loss': val_loss, 'vaild/f1_score' : valid_f1})

            if config_train['earlystopping']:
                earlystopping(valid_f1, model)
                if earlystopping.early_stop:
                    print("Early stopping")
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_train', type=str, help='path of train configuration yaml file')

    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    train(config_train)