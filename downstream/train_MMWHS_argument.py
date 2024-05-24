import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mydataset import mySegmentationDataset, SegmentationDataset
from utils import dice_loss
from evaluate import evaluate, evaluate_3d_iou
#from models.segmentation import UNet
import segmentation_models_pytorch as smp
import numpy as np
num_classes = 8
np.random.seed(42)

def train_net(net,
              device,
              epochs: int = 30,
              train_batch_size: int = 128,
              val_batch_size: int = 128,
              learning_rate: float = 0.1,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              out_dir : str= './checkpoint/'):
    # 1. Create dataset
    if args.dataset == "mmwhs_mri":
        train_dir_root = Path('/path/to/MMWHS_MR_Heart/')
        val_dir_img = Path('/path/to/MMWHS_MR_Heart/valid/')
        val_dir_mask = Path('/path/to/MMWHS_MR_Heart/valid_labels')
        test_dir_img = Path('/path/to/MMWHS_MR_Heart/test/')
        test_dir_mask = Path('/path/to/MMWHS_MR_Heart/test_labels')
        non_label_text = './non_labelMR_new.txt'
        have_label_text = './have_labelMR_new.txt'

    elif args.dataset == "mmwhs_ct":
        train_dir_root = Path('/path/to/MMWHS_CT_Heart/')
        val_dir_img = Path('/path/to/MMWHS_CT_Heart/valid/')
        val_dir_mask = Path('/path/to/MMWHS_CT_Heart/valid_labels')
        test_dir_img = Path('/path/to/MMWHS_CT_Heart/test/')
        test_dir_mask = Path('/path/to/MMWHS_CT_Heart/test_labels')
        non_label_text = './non_label.txt'
        have_label_text = './have_label.txt'

    dir_checkpoint = Path(out_dir)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    # non_label_text = '/home/hnguyen/KOTORI/Universal_downstream_code/non_label.txt'
    # have_label_text = '/home/hnguyen/KOTORI/Universal_downstream_code/have_label.txt'
    
    train_dataset = mySegmentationDataset(root_dir= train_dir_root, nonlabel_path= non_label_text, havelabel_path= have_label_text, dataset = args.dataset, scale= img_scale)
    # print(torch.unique(train_dataset[0]['mask']))
    val_dataset = SegmentationDataset(val_dir_img, val_dir_mask, img_scale, datasetname=args.dataset)
    # print(torch.unique(val_dataset[0]['mask']))
    test_dataset = SegmentationDataset(test_dir_img, test_dir_mask, img_scale, datasetname=args.dataset)
    # print(torch.unique(test_dataset[0]['mask']))    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    # exit(0)
    # 2. Split into train / validation partitions
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, **loader_args)
    import time 
    #start = time.time()
    #a= next(iter(train_loader))
    #end=time.time()
    #print(end-start)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, batch_size=val_batch_size,  **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, train_batch_size=train_batch_size, val_batch_size=val_batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Train batch size:      {train_batch_size}
        Val batch size: {val_batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
    print(learning_rate)
    # optimizer= optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_3d_iou = 0 
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                #true_masks[true_masks == 4] = 3
                #true_masks[true_masks >4] = 0
                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                clip_value = 1
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if global_step % (n_train // (1 * train_batch_size)) == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_dice_score, val_iou_score = evaluate(net, val_loader, device, 2)
                    val_3d_iou_score = evaluate_3d_iou(net, val_dataset, device, 2)

                    test_dice_score, test_iou_score = evaluate(net, test_loader, device, 2)
                    test_3d_iou_score = evaluate_3d_iou(net, test_dataset, device, 2)

                    # scheduler.step(val_dice_score)
                    if val_3d_iou_score  > best_3d_iou:
                        best_3d_iou = val_3d_iou_score
                        logging.info("New best 3d iou score: {}".format(best_3d_iou))
                        torch.save(net.state_dict(), str(dir_checkpoint/'checkpoint_{}_{}_best.pth'.format(args.dataset, args.pretrained)))

                    logging.info('Validation Dice score: {}, IoU score {}, IoU 3d score {}'.format(val_dice_score, val_iou_score, val_3d_iou_score))
                    logging.info('Testing Dice score: {}, IoU score {}, IoU 3d score {}'.format(test_dice_score, test_iou_score, test_3d_iou_score))
                    # img = torch.softmax(masks_pred, dim=1)[0].float().cpu()

        # Evaluation the last model
        if epoch + 1 == epochs:
            val_dice_score, val_iou_score = evaluate(net, val_loader, device, 2)
            val_3d_iou_score = evaluate_3d_iou(net, val_dataset, device, 2)

            test_dice_score, test_iou_score = evaluate(net, test_loader, device, 2)
            test_3d_iou_score = evaluate_3d_iou(net, test_dataset, device, 2)

            logging.info('Validation Dice score: {}, IoU score {}, IoU 3d score {}'.format(val_dice_score, val_iou_score, val_3d_iou_score))
            logging.info('Testing Dice score: {}, IoU score {}, IoU 3d score {}'.format(test_dice_score, test_iou_score, test_3d_iou_score))

        if save_checkpoint:
            # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    logging.info("Evalutating on test set")
    logging.info("Loading best model on validation")
    net.load_state_dict(torch.load(str(dir_checkpoint/'checkpoint_{}_{}_best.pth'.format(args.dataset, args.pretrained))))
    test_dice, test_iou = evaluate(net, test_loader, device, 2)
    test_3d_iou = evaluate_3d_iou(net, test_dataset, device, 2)
    logging.info("Test dice score {}, IoU score {}, 3d IoU {}".format(test_dice, test_iou, test_3d_iou))

    logging.info("Loading model at last epochs %d" %epochs)
    net.load_state_dict(torch.load(str(dir_checkpoint/'checkpoint_epoch{}.pth'.format(epochs))))
    test_dice_last, test_iou_last = evaluate(net, test_loader, device, 2)
    test_3d_iou_last = evaluate_3d_iou(net, test_dataset, device, 2)
    logging.info("Test dice score {}, IoU score {}, 3d IoU {}".format(test_dice_last, test_iou_last, test_3d_iou_last))

    return test_dice, test_iou, test_3d_iou, test_dice_last, test_iou_last, test_3d_iou_last

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train-batch-size', '-tb', dest='train_batch_size', metavar='TB', type=int, default=32, help='Batch size')
    parser.add_argument("--dataset", "-ds", dest="dataset", type=str, default="bts", help="choose dataset to run")
    parser.add_argument("--outdir", "-od", dest="outdir", type=str, default="./checkpoint", help="choose output directory")
    parser.add_argument('--val-batch-size', '-vb', dest='val_batch_size', metavar='VB', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument("--pretrained", "-pt", dest="pretrained", type=str, default="", help="Pretrained Resnet weights")
    parser.add_argument('--load', '-f', type=str, default=False, help='Load pretrained model from a checkpoint file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument("--workers", "-w", type=int, default=4, help="numbes of workers for DataLoader")
    parser.add_argument("--GPUs", "-gpu", type=str, default='cuda', help="Cuda position")
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cuda_string = 'cuda:' + args.GPUs
    device = torch.device(cuda_string if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    try:
        _2d_dices = []
        _2d_ious = []
        _3d_ious = []
        _2d_dices_last = []
        _2d_ious_last = []
        _3d_ious_last = []


        for trial in range(5):
            print ("----"*3)
            if args.pretrained == "scratch":
                net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=1, classes=num_classes)
            else:
                print ("Using pre-trained models from", args.pretrained)
                net = smp.Unet(encoder_name="resnet50", encoder_weights=args.pretrained ,in_channels=1, classes=num_classes)

            # logging.info(f'Network:\n'
            #              f'\t{net.n_cls} classes\n'
            #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

            if args.load:
                net.load_state_dict(torch.load(args.load, map_location=device))
                # net.load_state_dict(torch.load(args.path, map_location=device))
                logging.info(f'Model loaded from {args.load}')
            net.to(device=device)

            print("Trial", trial + 1)
            _2d_dice, _2d_iou, _3d_iou, _2d_dice_last, _2d_iou_last, _3d_iou_last = train_net(net=net,
                    epochs=args.epochs,
                    train_batch_size=args.train_batch_size,
                    val_batch_size=args.val_batch_size,
                    learning_rate=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100,
                    amp=args.amp,
                    out_dir= args.outdir)
            _2d_dices.append(_2d_dice.item())
            _2d_ious.append(_2d_iou.item())
            _3d_ious.append(_3d_iou.item())
            _2d_dices_last.append(_2d_dice_last.item())
            _2d_ious_last.append(_2d_iou_last.item())
            _3d_ious_last.append(_3d_iou_last.item())

        print ("Average performance on best valid set")
        print("2d dice {}, mean {}, std {}".format(_2d_dices, np.mean(_2d_dices), np.std(_2d_dices)))
        print("2d iou {}, mean {}, std {}".format(_2d_ious, np.mean(_2d_ious), np.std(_2d_ious)))
        print("3d iou {}, mean {}, std {}".format(_3d_ious, np.mean(_3d_ious), np.std(_3d_ious)))

        print ("Average performance on the last epoch")
        print("2d dice {}, mean {}, std {}".format(_2d_dices_last, np.mean(_2d_dices_last), np.std(_2d_dices_last)))
        print("2d iou {}, mean {}, std {}".format(_2d_ious_last, np.mean(_2d_ious_last), np.std(_2d_ious_last)))
        print("3d iou {}, mean {}, std {}".format(_3d_ious_last, np.mean(_3d_ious_last), np.std(_3d_ious_last)))

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
