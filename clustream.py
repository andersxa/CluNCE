import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import erfinv
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import numpy as np
import wandb

import os
wandb_off = os.getenv('WANDB_MODE') == 'disabled'

class CluStream(nn.Module):
    def __init__(self, args, loader, model_fn):
        super().__init__()
        self.initialized = False
        self.model = model_fn()
        self.microclusters = []
        self.num_initial_microclusters = args.num_initial_microclusters
        self.latent_dim = args.latent_dim
        self.temperature = args.temperature
        self.delta_timestamp = args.delta_timestamp
        self.m_recent = args.m_recent
        self.max_boundary_factor = args.max_boundary_factor
        self.t = 1
    
    def get_state_dict(self):
        return {
            'model': self.model.state_dict(),
            'microclusters': self.microclusters,
            'num_initial_microclusters': self.num_initial_microclusters,
            'latent_dim': self.latent_dim,
            'temperature': self.temperature,
            'delta_timestamp': self.delta_timestamp,
            'm_recent': self.m_recent,
            'max_boundary_factor': self.max_boundary_factor,
            't': self.t,
            'initialized': self.initialized
        }

    @torch.no_grad()
    def set_closest_microcluster(self):
        if not any([mc.n < 5 for mc in self.microclusters]):
            return
        if any([mc.n >= 5 for mc in self.microclusters]):
            # For each microcluster, set the maximum boundary to the distance to the nearest other microcluster
            above_clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters if mc.n >= 5])
            five_clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters if mc.n < 5])
            all_clusters = torch.cat((five_clusters, above_clusters), dim=0)
        else:
            all_clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters])
            five_clusters = all_clusters
        distances = torch.cdist(five_clusters, all_clusters)
        distances.fill_diagonal_(float('inf'))
        less_than_5 = [mc for mc in self.microclusters if mc.n < 5]
        wandb.log({'cluster_sizes': [mc.n for mc in self.microclusters], 'cluster_times': [mc.t1 / mc.n for mc in self.microclusters]}, commit=False)
        #print('Sizes', len(less_than_5), distances.shape, five_clusters.shape, all_clusters.shape, len(self.microclusters))
        for i, mc in enumerate(less_than_5):
            mc.closest_microcluster = torch.amin(distances[i, :])

    @torch.no_grad()
    def update_from_loader(self, args, loader, device):
        if self.initialized:
            return
        self.initialized = True
        # Perform k-means clustering on the latent space
        self.model.eval()
        #First obtain the latents
        latents = []
        print("Obtaining latents for k-means clustering")
        for i, (x_k, _) in tqdm(enumerate(loader), total=min(args.num_update_batches, len(loader)), unit_scale=args.test_batch_size, disable=not wandb_off):
            x_k = x_k.to(device)
            z = self.model(x_k)
            latents.append(z.detach().cpu())
            if i >= args.num_update_batches:
                break
        latents = torch.cat(latents, dim=0).numpy()
        print(f'Performing k-means clustering on latents of shape {latents.shape}')
        #Then perform k-means clustering
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_initial_microclusters, n_init='auto', max_iter=args.max_iter)
        labels = self.kmeans.fit_predict(latents)
        c_idx, counts = np.unique(labels, return_counts=True)

        for i, (c, count) in enumerate(zip(c_idx, counts)):
            mc = MicroCluster(i, self.latent_dim, t=self.max_boundary_factor)
            mc_centroid = torch.tensor(self.kmeans.cluster_centers_[c], device='cuda')
            mc.init_from_latent(mc_centroid, self.t)
            mc.x1 = mc.x1 * count
            mc.x2 = mc.x2 * count
            mc.t1 = mc.t1 * count
            mc.t2 = mc.t2 * count
            mc.n = count
            self.microclusters.append(mc)
        if len(counts) < self.num_initial_microclusters:
            for i in range(len(counts), self.num_initial_microclusters):
                mc = MicroCluster(i, self.latent_dim, t=self.max_boundary_factor)
        self.set_closest_microcluster()
        self.cur_i = self.num_initial_microclusters
        print("Finished k-means clustering")

    def encode(self, x):
        x = self.model(x)
        clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters])
        distances = torch.cdist(x, clusters)
        closest_microcluster = torch.argmin(distances, dim=1)
        x = F.normalize(x, dim=1)
        logits = x @ clusters.t()
        loss = F.cross_entropy(logits / self.temperature, closest_microcluster)
        return x, closest_microcluster, loss
        
    def forward(self, x_q, x_k):
        self.t += 1
        #x_q and x_k are just augmentations of the same image
        #First encode x_q and x_k
        z_q = self.model(x_q)
        z_k = self.model(x_k)

        #Clusters
        clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters])
        #Distances
        distances = torch.cdist(z_q, clusters)
        #Find the closest microcluster
        closest_microcluster = torch.argmin(distances, dim=1)
        if not self.training:
            z_k = F.normalize(z_k, dim=1)
            logits = z_k @ clusters.t()
            loss = F.cross_entropy(logits / self.temperature, closest_microcluster)
            return z_k, closest_microcluster, loss
        closest_microcluster_dist = torch.amin(distances, dim=1)
        candidates_for_deletion = []
        #Maintenance. First decide if it is possible to delete a microcluster
        #Estimate the average timestamp of the last m arrivals,
        #The microcluster with the least recent timestamp can be deleted if it is older than delta_timestamp
        #We have the sum of timestamps and the sum of squared timestamps, and the number of points
        #Assuming a normal distribution, we can find the time of arrival of the m/(2*n)th percentile of the points in the microcluster
        
        for i, mc in enumerate(self.microclusters):
            if self.m_recent > 2 * mc.n:
                continue
            #mu:
            mu = mc.t1 / mc.n
            #std:
            std = math.sqrt((mc.t2 / mc.n) - mu**2)
            #m/(2*n)th percentile using the inverse CDF of the normal distribution
            percentile = 2 * (self.m_recent / (2 * mc.n)) - 1
            stamp = erfinv(percentile) * std + mu
            
            deletable = stamp < self.delta_timestamp * self.t
            #Deletable only if not closest to any data point
            if deletable and not torch.any(closest_microcluster == i):
                #print('FOR_DELETION', stamp, self.delta_timestamp, self.t, mc.t1 / mc.n, self.m_recent, mc.n)
                candidates_for_deletion.append((mc, stamp, i))
        
        candidates_for_deletion.sort(key=lambda x: x[1])
        
        #Add the latent vector to the closest microcluster
        new_clusters = False
        microcluster_dists = None
        labels = []
        for z, m, d in zip(z_q, closest_microcluster, closest_microcluster_dist):
            mc = self.microclusters[m]
            if mc.add_decision(d):
                mc.add(z, self.t)
                labels.append(mc.id)
            else:
                #First if there are clusters for deletion, delete them and create a new cluster for this point
                if len(candidates_for_deletion) > 0:
                    mc_del, _, mc_idx = candidates_for_deletion.pop()
                    mc_del.deleted = True
                    if microcluster_dists is not None:
                        microcluster_dists[mc_idx, :] = float('inf')
                        microcluster_dists[:, mc_idx] = float('inf')
                else:
                    #Merge closest two microclusters
                    if microcluster_dists is None:
                        microcluster_dists = torch.cdist(clusters, clusters)
                        microcluster_dists.fill_diagonal_(float('inf'))
                        #Not any closest microclusters to current batch
                        microcluster_dists[closest_microcluster, :] = float('inf')
                        microcluster_dists[:, closest_microcluster] = float('inf')
                    #Find the closest two microclusters
                    row, col = divmod(microcluster_dists.argmin().item(), microcluster_dists.shape[1])
                    #Merge them
                    mc1 = self.microclusters[row]
                    mc2 = self.microclusters[col]
                    if mc1.deleted or mc2.deleted:
                        microcluster_dists[row, col] = float('inf')
                        labels.append(-1)
                        continue
                    mc1.x1 = mc1.x1 + mc2.x1
                    mc1.x2 = mc1.x2 + mc2.x2
                    mc1.t1 = mc1.t1 + mc2.t1
                    mc1.t2 = mc1.t2 + mc2.t2
                    mc1.n = mc1.n + mc2.n
                    mc2.deleted = True
                    #Update distances
                    microcluster_dists[row, :] = float('inf')
                    microcluster_dists[:, row] = float('inf')
                    microcluster_dists[col, :] = float('inf')
                    microcluster_dists[:, col] = float('inf')
                    #Set the closest microcluster
                    mc1.closest_microcluster = torch.amin(microcluster_dists[row, :])
                    #Check if mc2 is in candidates for deletion
                    for i, (mc, _, _) in enumerate(candidates_for_deletion):
                        if mc.id == mc2.id:
                            candidates_for_deletion.pop(i)
                            break                    

                self.cur_i += 1
                new_mc = MicroCluster(self.cur_i, self.latent_dim, t=self.max_boundary_factor)
                new_mc.init_from_latent(z, self.t)
                self.microclusters.append(new_mc)
                new_clusters = True
                labels.append(new_mc.id)

        #Delete clusters
        self.microclusters = [mc for mc in self.microclusters if not mc.deleted]
        id_map = defaultdict(lambda: -1, {mc.id: i for i, mc in enumerate(self.microclusters)})

        #Update labels to match new microclusters
        labels = torch.tensor([id_map[l] for l in labels], device='cuda')

        #Calculate loss
        clusters = torch.stack([mc.x1 / mc.n for mc in self.microclusters])
        clusters = F.normalize(clusters, dim=1)
        z_k = F.normalize(z_k, dim=1)
        logits = z_k @ clusters.t()
        loss = F.cross_entropy(logits / self.temperature, labels, ignore_index=-1)
        
        if new_clusters:
            self.set_closest_microcluster()
        
        #Detach all microclusters
        for mc in self.microclusters:
            mc.x1 = mc.x1.detach()
            mc.x2 = mc.x2.detach()
        
        return z_k, labels, loss

class MicroCluster:
    def __init__(self, id, latent_dim, t=6.0, device='cuda'):
        self.id = id
        self.x1 = torch.randn(latent_dim, device='cuda') #Sum of latent vectors
        self.x2 = self.x1.square() #Sum of squared latent vectors
        self.t1 = 1 #Sum of timestamps (here timesteps are constant)
        self.t2 = 1 #Sum of squared timestamps
        self.n = 1
        self.closest_microcluster = None #Set by parent CluStream object
        self.t = t # Maximum boundary factor
        self.deleted = False

    def init_from_latent(self, z, t):
        self.x1 = z
        self.x2 = z.square()
        self.t1 = t
        self.t2 = t ** 2
        self.n = 1
    
    @torch.no_grad()
    def add_decision(self, d):
        #This function is called when x is closest to this microcluster
        #Only add x if it is within the maximum boundary times a factor t
        #Use t * RMS deviation of the microcluster if n > 5, otherwise use the distance to the closest microcluster
        if self.n < 5:
            max_boundary = self.closest_microcluster
        else:
            centroid = self.x1 / self.n
            rms_deviation = torch.sqrt(torch.mean(self.x2 / self.n - centroid.square()))
            max_boundary = self.t * rms_deviation
        if d < max_boundary:
            #print(f'Max boundary: {max_boundary}, distance: {d}, n: {self.n}, closest: {self.closest_microcluster}')
            return True
        else:
            return False
        
    def add(self, x, t):
        self.x1 += x
        self.x2 += x.square()
        self.t1 += t
        self.t2 += t ** 2
        self.n += 1