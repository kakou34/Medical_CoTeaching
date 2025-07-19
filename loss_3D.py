import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, train_loader):

    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    print( f"num_remember :{num_remember}, len y1: {len(y_1)}, len y2: {len(y_2)}")


    # Select indices for the top `num_remember` points from each sorted array
    selected_indices_1 = ind[ind_1_sorted[:num_remember]]
    selected_indices_2 = ind[ind_2_sorted[:num_remember]]


    non_noisy_count_1 = np.sum(noise_or_not[selected_indices_1])
    non_noisy_count_2 = np.sum(noise_or_not[selected_indices_2])


    # Compute the pure ratios
    pure_ratio_1 = non_noisy_count_1 / float(num_remember)
    pure_ratio_2 = non_noisy_count_2 / float(num_remember)

    # Identify discarded samples (high-loss samples)
    discarded_indices_1 = set(ind[ind_1_sorted[num_remember:].cpu()].tolist())  
    discarded_indices_2 = set(ind[ind_2_sorted[num_remember:].cpu()].tolist())

    #discarded samples per epoch 
    latest_discarded = {}
    discarded_samples_per_epoch = []

    for idx in discarded_indices_1.union(discarded_indices_2):  
        ptid = train_loader.dataset.data[idx]  # Retrieve PTID
        noise_status = int(noise_or_not[idx].cpu().item()) # 1 = clean, 0 = noise
        latest_discarded[ptid] = {
            "PTID": ptid,
            "Noise_Status": noise_status,
            "Discarded_Model1": idx in discarded_indices_1,  
            "Discarded_Model2": idx in discarded_indices_2  
        }

    # Save last decision per sample 
    discarded_samples_per_epoch = list(latest_discarded.values())

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return loss_1, loss_2, pure_ratio_1, pure_ratio_2, torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, discarded_samples_per_epoch


