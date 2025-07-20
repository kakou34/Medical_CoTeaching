import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]).item() / float(num_remember)
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]).item() / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    # 🛠️ ➔ Hier geef ik de originele per-sample losses terug
    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
        pure_ratio_1,
        pure_ratio_2,
        loss_1.detach(),
        loss_2.detach()
    )


if __name__=="__main__":
    y1 = torch.tensor([[0.3, 0.7],[0.8, 0.2],[0.4, 0.6],[0.1, 0.9],[0.01, 0.99]])
    
    y2 = torch.tensor([[0.4, 0.6],[0.1, 0.9],[0.5, 0.5],[0.2, 0.8],[0.9, 0.1]])

    t = torch.tensor([1,0,0,1,0])
    forget_rate = 0.5
    ind = torch.tensor([0,1,2,3,4])
    noise_not = torch.tensor([0,0,0,1,1])


    l1, l2, pure_r1, pure_r2 = loss_coteaching(y1, y2, t, forget_rate, ind, noise_not)

    print(l1)
    print(l2)
    print(pure_r1)
    print(pure_r2)
