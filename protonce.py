# Reimplementation (from scratch) and edits of https://github.com/salesforce/PCL/
# https://arxiv.org/abs/2005.04966
# Li, Junnan, et al. "Prototypical contrastive learning of unsupervised representations." arXiv preprint arXiv:2005.04966 (2020).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from collections import defaultdict

import os
wandb_off = os.getenv('WANDB_MODE') == 'disabled'

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class ProtoNCE(nn.Module):
    def __init__(self, args, loader, model_fn):
        super().__init__()
        self.model_k = model_fn()
        self.model_q = model_fn()

        #Copy weights from q into k
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False # k is momentum encoder, do not accumulate gradients
        
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.temperature = args.temperature

        self.register_buffer("queue", torch.randn(args.latent_dim, self.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_cursor = 0

        self.register_buffer("prototypes", torch.zeros(args.latent_dim, args.num_prototypes))
        self.register_buffer("densities", torch.zeros(args.num_prototypes))
        self.kmeans = None

    def get_state_dict(self):
        return {
            'model_k': self.model_k.state_dict(),
            'model_q': self.model_q.state_dict(),
            'queue': self.queue,
            'queue_cursor': self.queue_cursor,
            'prototypes': self.prototypes,
            'densities': self.densities,
            'kmeans': self.kmeans,
            'queue_size': self.queue_size,
            'momentum': self.momentum,
            'temperature': self.temperature
        }

    def encode(self, x):
        x = self.model_k(x)
        x = F.normalize(x, dim=1)
        #Prototype loss
        prototype_logits = x @ self.prototypes
        prototype_logits = prototype_logits / self.densities.unsqueeze(0)
        closest_prototypes = prototype_logits.argmax(dim=1)
        prototype_loss = F.cross_entropy(prototype_logits, closest_prototypes)
        return x, closest_prototypes, prototype_loss

    def k_encode(self, x):
        x = self.model_k(x)
        x = F.normalize(x, dim=1)
        return x
    
    def q_encode(self, x):
        x = self.model_q(x)
        x = F.normalize(x, dim=1)
        return x

    @torch.no_grad()
    def update_from_loader(self, args, loader, device):
        # Perform k-means clustering on the latent space
        self.model_k.eval()
        #First obtain the latents
        latents = []
        print("Obtaining latents for k-means clustering")
        for i, (x_k, _) in tqdm(enumerate(loader), total=min(args.num_update_batches, len(loader)), unit_scale=args.test_batch_size, disable=not wandb_off):
            x_k = x_k.to(device)
            z = self.k_encode(x_k)
            latents.append(z.detach().cpu())
            if i >= args.num_update_batches:
                break
        latents = torch.cat(latents, dim=0).numpy()
        print(f'Performing k-means clustering on latents of shape {latents.shape}')
        #Then perform k-means clustering
        self.kmeans = KMeans(n_clusters=self.prototypes.shape[1], n_init=args.n_init, max_iter=args.max_iter)
        labels = self.kmeans.fit_predict(latents)
        #Update prototypes
        prototypes = torch.tensor(self.kmeans.cluster_centers_)
        prototypes = F.normalize(prototypes, dim=1).t()
        self.prototypes.copy_(prototypes)
        #Distances
        distances = pairwise_distances(latents, self.kmeans.cluster_centers_)
        #Cluster distances and densities - density estimation is from the paper
        cluster_distances = defaultdict(list)
        densities = torch.zeros(args.num_prototypes)
        for point, cluster in enumerate(labels):
            cluster_distances[cluster].append(distances[point, cluster])
        for cluster in cluster_distances:
            n_assignments = len(cluster_distances[cluster])
            if n_assignments > 1:
                dists = torch.tensor(cluster_distances[cluster]).mean()
                densities[cluster] = dists / np.log(n_assignments+10) #from paper

        max_density = densities.max()
        for cluster in cluster_distances:
            if densities[cluster] == 0:
                densities[cluster] = max_density
        densities = densities.clamp(min=torch.quantile(densities, 0.1), max=torch.quantile(densities, 0.9))
        densities = args.temperature * densities / densities.mean()
        self.densities.copy_(densities)
        print("Finished k-means clustering")

    def forward(self, x_q, x_k):
        with torch.no_grad():
            if self.training:
                #Momentum step
                for p_q, p_k in zip(self.model_q.parameters(), self.model_k.parameters()):
                    p_k.data = self.momentum * p_k.data + (1.0 - self.momentum) * p_q.data
            
            #Get key latents
            k = self.k_encode(x_k)

        #Get query latents
        q = self.q_encode(x_q)

        #Get positive and negative logits
        logits_positive = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        logits_negative = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([logits_positive, logits_negative], dim=1) / self.temperature

        #Cross-entropy loss
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        instance_loss = F.cross_entropy(logits, labels)

        if self.training:
            #Update queue
            self.update_queue(k)

        #Prototype loss
        prototype_logits = q @ self.prototypes
        prototype_logits = prototype_logits / self.densities.unsqueeze(0)
        closest_prototypes = prototype_logits.argmax(dim=1)
        prototype_loss = F.cross_entropy(prototype_logits, closest_prototypes)

        return q, closest_prototypes, prototype_loss + instance_loss


    def update_queue(self, z):
        batch_size = z.shape[0]
        z = z.detach()
        self.queue[:, self.queue_cursor:self.queue_cursor+batch_size] = z.t()
        self.queue_cursor = (self.queue_cursor + batch_size) % self.queue_size
        