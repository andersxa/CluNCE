import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from protonce import ProtoNCE, TwoCropsTransform
from clustream import CluStream

from torch.utils.data import DataLoader
#datasets either fashion_mnist or ImageNet
from torchvision import datasets, transforms
from imagenet_dataset import ImageNet
from torchvision.models import resnet50, resnet101, vgg16, resnet18

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.cluster import MiniBatchKMeans

import wandb
import os
wandb_off = os.getenv('WANDB_MODE') == 'disabled'


import numpy as np
from tqdm import tqdm

import argparse

#CluStream is called per-batch while ProtoNCE is called per-epoch
#Everything is unsupervised, so discard the labels
def train_epoch(args, device, train_loader, update_loader, test_loader, optimizer, epoch, cluster_obj):
    with tqdm(enumerate(train_loader), total=len(train_loader), unit_scale=args.batch_size, disable=not wandb_off) as progress_bar:
        for i, ((x_q, x_k), l) in progress_bar:
            cluster_obj.train()
            x_q = x_q.to(device)
            x_k = x_k.to(device)
            optimizer.zero_grad()
            z, c_preds, loss = cluster_obj(x_q, x_k)
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                train_ami = adjusted_mutual_info_score(l.cpu().numpy(), c_preds.cpu().numpy())
                #progress_bar.set_description(f'Epoch {epoch} Loss: {loss.item():.6f} AMI: {train_ami:.4f}')
                wandb.log({"train_loss": loss.item(), "train_ami": train_ami, "epoch": epoch+float(i)/len(train_loader)})
            if i % args.eval_interval == 0:
                optimizer.zero_grad()
                test(args, device, test_loader, epoch+float(i)/len(train_loader), cluster_obj)
    optimizer.zero_grad()
    cluster_obj.update_from_loader(args, update_loader, device)
    
def kmeans_adjusted_mutual_info_score(labels, latents):
    print("Performing k-means clustering (k=25000) on latents of shape", latents.shape)
    kmeans = MiniBatchKMeans(n_clusters=25000, batch_size=8192, n_init='auto')
    kmeans.fit(latents)
    print("Finished k-means clustering")
    return adjusted_mutual_info_score(labels, kmeans.labels_)

def test(args, device, test_loader, epoch, cluster_obj):
    cluster_obj.eval()
    with torch.no_grad():
        test_loss = torch.tensor(0.0, device=device)
        latents = []
        labels = []
        prototype_preds = []
        for x_k, l in tqdm(test_loader, disable=not wandb_off):
            x_k = x_k.to(device)
            z, c_preds, loss = cluster_obj.encode(x_k)
            test_loss += loss # sum up batch loss
            latents.append(z)
            labels.append(l)
            prototype_preds.append(c_preds)
        latents = torch.cat(latents, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        prototype_preds = torch.cat(prototype_preds, dim=0).cpu().numpy()
        test_loss = test_loss.cpu().item()
    prototypical_ami = adjusted_mutual_info_score(labels, prototype_preds)
    #kmeans_ami = kmeans_adjusted_mutual_info_score(labels, latents)
    test_loss /= len(test_loader)
    wandb.log({"test_loss": test_loss, "test_ami": prototypical_ami, "epoch": epoch})
    #print(f"Test set: Average loss: {test_loss:.4f}, Prototypical AMI: {prototypical_ami:.4f}") #, K-Means AMI: {kmeans_ami:.4f}")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Clustering')
    parser.add_argument('--clustering', type=str, default='protonce',
                        help='clustering method to use (default: protonce)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)') #64
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)') #1000
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)') #20
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)') #0.01
    parser.add_argument('--seed', type=int, default=924813484, metavar='S',
                        help='random seed (default: 924813484)') #924813484
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status') #100
    parser.add_argument('--eval-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before evaluating model') #1000
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model') #True
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('--model-save-path', type=str, default='models/',
                        help='Path to save the model (default: models/)')
    parser.add_argument('--data-path', type=str, default='data/',
                        help='Path to save the data (default: data/)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for data loading (default: 16)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model to use (default: resnet18)') #resnet50, resnet101 or vgg16 - or if fashion_mnist then resnet18
    
    #Clustering arguments
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension (default: 128)') #128
    parser.add_argument('--temperature', type=float, default=0.2, metavar='T',
                        help='temperature for softmax (default: 0.2)') #0.2
    parser.add_argument('--num-update-batches', type=int, default=100,
                        help='Number of batches to use for update (default: 100)')
    parser.add_argument('--max-iter', type=int, default=30,
                        help='Maximum number of k-means iterations (default: 30)')
    #ProtoNCE
    parser.add_argument('--queue-size', type=int, default=8192,
                        help='Size of queue (default: 8192)')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='Momentum for queue update (default: 0.999)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Number of warmup epochs (default: 3)')
    parser.add_argument('--num-prototypes', type=int, default=300,
                        help='Number of prototypes (default: 300)')
    parser.add_argument('--n-init', type=int, default=5,
                        help='Number of k-means initializations (default: 5)')
    #CluStream
    parser.add_argument('--num-initial-microclusters', type=int, default=8192,
                        help='Number of micro-clusters (default: 8192)')
    parser.add_argument('--delta-timestamp', type=float, default=0.01,
                        help='Time threshold for micro-cluster deletion (default: 0.01)')
    parser.add_argument('--m-recent', type=int, default=20,
                        help='Number of recent points before deletion (default: 20)')
    parser.add_argument('--max-boundary-factor', type=float, default=2.0,
                        help='Maximum boundary factor (default: 2.0)')
    
    

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA not available. Exiting...")
        exit()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.dataset.lower() == 'fashion_mnist' and args.model.lower() != 'resnet18':
        print("Fashion MNIST only supports ResNet18. Exiting...")
        exit()

    config = vars(args)
    wandb.init(project="CluNCE", config=config)

    if args.dataset.lower() == 'fashion_mnist':
        train_loader = DataLoader(
            datasets.FashionMNIST(args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        test_loader = DataLoader(
            datasets.FashionMNIST(args.data_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
    elif args.dataset.lower() == 'imagenet':

        #MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709 and also used in ProtoNCE
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])  
        train_loader = DataLoader(
            ImageNet(args.data_path + 'ImageNet/', split='train', transform=TwoCropsTransform(train_transform)),
            batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        update_loader = DataLoader(
            ImageNet(args.data_path + 'ImageNet/', split='train', transform=eval_transform),
            batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)
        test_loader = DataLoader(
            ImageNet(args.data_path + 'ImageNet/', split='val', transform=eval_transform),
            batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)

    model_fn = None
    if args.model.lower() == 'resnet50':
        model_fn = lambda: resnet50(num_classes=args.latent_dim)
    elif args.model.lower() == 'resnet101':
        model_fn = lambda: resnet101(num_classes=args.latent_dim)
    elif args.model.lower() == 'vgg16':
        model_fn = lambda: vgg16(num_classes=args.latent_dim)
    elif args.model.lower() == 'resnet18':
        def get_resnet18_fashion_mnist_model():
            m = resnet18(num_classes=args.latent_dim)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            return m
        model_fn = get_resnet18_fashion_mnist_model
    else:
        print("Unknown model. Exiting...")
        exit()
    
    cluster_obj = None
    if args.clustering.lower() == 'protonce':
        if args.queue_size % args.batch_size != 0:
            print("Queue size must be divisible by batch size. Exiting...")
            exit()
        cluster_obj = ProtoNCE(args, train_loader, model_fn).to(device)
        cluster_obj.update_from_loader(args, update_loader, device)
    elif args.clustering.lower() == 'clustream':
        cluster_obj = CluStream(args, train_loader, model_fn).to(device)
        cluster_obj.update_from_loader(args, update_loader, device)
    else:
        print("Unknown clustering method. Exiting...")
        exit()
    
    optimizer = optim.AdamW(cluster_obj.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, device, train_loader, update_loader, test_loader, optimizer, epoch, cluster_obj)
        test(args, device, test_loader, epoch+1.0, cluster_obj)
        if args.save_model:
            torch.save(cluster_obj.get_state_dict(), args.model_save_path + str(wandb.run.id) + "_" + str(wandb.run.name) + "_" + args.clustering.lower() + "_" + args.model.lower() + "_" + str(epoch) + ".pt")
    
if __name__ == '__main__':
    main()