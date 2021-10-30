import os
import yaml
import argparse
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), config_infer['model'])
    model = model_cls(
        num_classes=num_classes,
        freeze = True
    )

    model = torch.nn.DataParallel(model)

    model_path = os.path.join(saved_model, 'resnet50-cutmix30%.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(config_infer):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(config_infer['model_dir'], num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(config_infer['data_dir'], 'cropped_images')
    info_path = os.path.join(config_infer['data_dir'], 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    # -- define dataset
    dataset1 = TestDataset(img_paths, trans_n=1)
    dataset2 = TestDataset(img_paths, trans_n=2)
    dataset3 = TestDataset(img_paths, trans_n=3)

    # -- define dataloader
    loader1 = DataLoader(
        dataset1,
        num_workers=config_infer['num_workers'],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader2 = DataLoader(
        dataset2,
        num_workers=config_infer['num_workers'],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader3 = DataLoader(
        dataset3,
        num_workers=config_infer['num_workers'],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")

    # -- save prediction
    all_preds = []
    all_logits = np.zeros((1, 18))
    pred_1 = []
    pred_2 = []
    pred_3 = []

    # -- inference(TTA)
    with torch.no_grad():
        for _, images in enumerate(loader1):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            pred_1.extend((2*pred).cpu())
    with torch.no_grad():
        for _, images in enumerate(loader2):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            pred_2.extend(pred.cpu())
    with torch.no_grad():
        for _, images in enumerate(loader3):
            images = images.to(device)
            pred = model(images)
            all_logits = np.vstack([all_logits, pred.cpu()])
            pred_3.extend((2*pred).cpu())

    for k in range(len(pred_1)):
        npred = torch.zeros(images.size(0), 18)
        npred = torch.add(npred, pred_1[k])
        npred = torch.add(npred, pred_2[k])
        npred = torch.add(npred, pred_3[k])
        npred = npred.argmax(dim=-1) 

        all_preds.extend(npred.cpu().numpy())

    info['ans'] = all_preds
    info.to_csv(os.path.join(config_infer['output_dir'], f'output.csv'), index=False)
    np.save(os.path.join(config_infer['output_dir'], 'logit.npy'), all_logits)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_infer', type=str, help='path of inference configuration yaml file')
    args = parser.parse_args()

    with open(args.config_infer) as f:
        config_infer = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(config_infer['output_dir'], exist_ok=True)

    inference(config_infer)