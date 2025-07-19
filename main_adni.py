
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from monai.transforms import Compose, LoadImage, Resize, NormalizeIntensity, RandRotate, RandGaussianNoise, RandAdjustContrast, Rand3DElastic
from monai.networks.nets import resnet18
import matplotlib.pyplot as plt
import argparse
import numpy as np
from data.ADNI import ADNI 
from loss_3D import loss_coteaching  


# Adjust Learning Rate Function
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

# Accuracy Calculation
def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train Function
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    global discarded_samples
    discarded_samples = []
    per_sample_losses = [] 
    pure_ratio_1_list = []
    pure_ratio_2_list = []
    loss1_list = []
    loss2_list = []
    index_list = []
    noise_status_list = []
    epoch_loss1 = 0
    epoch_loss2 = 0
    
    for i, (images, labels, indexes) in enumerate(train_loader):
        if i > args.num_iter_per_epoch:
            break

        images = images.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        logits1 = model1(images)
        logits2 = model2(images)

         # Convert to probabilities
        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)

        # Check for NaNs in probabilities
        assert not torch.isnan(prob1).any(), "Model1 produced NaN probabilities!"
        assert not torch.isnan(prob2).any(), "Model2 produced NaN probabilities!"

        # Check if the predicted class matches the highest probability
        _, pred1 = torch.max(prob1, 1)
        _, pred2 = torch.max(prob2, 1)
        assert torch.all(pred1 == torch.argmax(prob1, dim=1)), "Model1 predictions do not match probabilities!"
        assert torch.all(pred2 == torch.argmax(prob2, dim=1)), "Model2 predictions do not match probabilities!"

        loss_1, loss_2,pure_ratio_1, pure_ratio_2, total_loss_1, total_loss_2, discarded_samples_per_epoch = loss_coteaching(
            logits1, logits2, labels, rate_schedule[epoch], indexes.cuda(), noise_or_not, train_loader
        ) 
       
 
        optimizer1.zero_grad()
        total_loss_1.backward()
        optimizer1.step()
        epoch_loss1 += total_loss_1.item()

        optimizer2.zero_grad()
        total_loss_2.backward()
        optimizer2.step()
        epoch_loss2 += total_loss_2.item()

        # Store only the latest discarded samples per PTID
        discarded_samples.extend(discarded_samples_per_epoch)
       
        # Retrieve PTIDs from the dataset
        ptids = [train_loader.dataset.data[idx] for idx in indexes.cpu().tolist()]
        noise_statuses = [noise_or_not[idx].item() for idx in indexes.cpu().tolist()]  # Extract noise status

         # Store PTID, loss per sample, and noise status
        for ptid, l1, l2, noise in zip(ptids, loss_1.detach().cpu().tolist(), loss_2.detach().cpu().tolist(), noise_statuses):
            per_sample_losses.append({
                "PTID": ptid,
                "Loss_Model1": l1,
                "Loss_Model2": l2,
                "Noise_Status": noise  # 1 = Clean, 0 = Noisy
            })
        print(f"Epoch [{epoch+1}/{args.n_epoch}], Iter [{i+1}/{len(train_loader)}] Loss1: {total_loss_1.item():.4f}, Loss2: {total_loss_2.item():.4f}")


        pure_ratio_1_list.append(pure_ratio_1.item() * 100)
        pure_ratio_2_list.append(pure_ratio_2.item() * 100)

        print(f"Epoch [{epoch+1}/{args.n_epoch}], Iter [{i+1}/{len(train_loader)}] Total Loss1: {total_loss_1.item():.4f}, Loss2: {total_loss_2.item():.4f}")
   

    epoch_loss1 /= len(train_loader)
    epoch_loss2 /= len(train_loader)

    # Convert per-sample loss data to DataFrame
    df_loss = pd.DataFrame(per_sample_losses)
    loss_csv_path = f"resultsCT/loss_per_sample_alz_noisetype_epoch_{epoch}.csv"
    df_loss.to_csv(loss_csv_path, index=False)
    print(f"Saved per-sample losses to {loss_csv_path}")
   
    mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
    mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

    return mean_pure_ratio1, mean_pure_ratio2, loss1_list, loss2_list, index_list, epoch_loss1, epoch_loss2, discarded_samples



# Evaluate Function
def evaluate(test_loader, model1, model2):
    model1.eval()
    model2.eval()
    correct1, correct2 = 0, 0
    total1, total2 = 0, 0

    with torch.no_grad():
        for images, labels, index in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            


            logits1 = model1(images)
            logits2 = model2(images)
             
            # Convert to probabilities
            prob1 = F.softmax(logits1, dim=1)
            prob2 = F.softmax(logits2, dim=1)

        # Check for NaNs in probabilities
            assert not torch.isnan(prob1).any(), "Model1 produced NaN probabilities!"
            assert not torch.isnan(prob2).any(), "Model2 produced NaN probabilities!"

        # Check if the predicted class matches the highest probability
            _, pred1 = torch.max(prob1, 1)
            _, pred2 = torch.max(prob2, 1)
            assert torch.all(pred1 == torch.argmax(prob1, dim=1)), "Model1 predictions do not match probabilities!"
            assert torch.all(pred2 == torch.argmax(prob2, dim=1)), "Model2 predictions do not match probabilities!"

            prob1 = F.softmax(logits1, dim=1)
            prob2 = F.softmax(logits2, dim=1)
            _, pred1 = torch.max(prob1, 1)
            _, pred2 = torch.max(prob2, 1)
           
            total1 += labels.size(0)
            correct1 += (pred1 == labels).sum().item()
            total2 += labels.size(0)
            correct2 += (pred2 == labels).sum().item()

    acc1 = 100 * correct1 / total1
    acc2 = 100 * correct2 / total2
    return acc1, acc2

def evaluatefinal(test_loader, model1, model2):
    model1.eval()
    model2.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            logits1 = model1(images)
            logits2 = model2(images)

            # Mean logits
            final_logits = (logits1 + logits2) / 2.0

            # Softmax probailities 
            probs = F.softmax(final_logits, dim=1)

            # save 
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    

    # calculate accuracy
    preds = torch.argmax(all_probs_tensor, dim=1)
    correct = (preds == all_labels_tensor).sum().item()
    acc_combined = 100.0 * correct / len(all_labels_tensor)
    return all_probs_tensor.numpy(), all_labels_tensor.numpy(), acc_combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--noise_rate', type=float, default= 0.10)
    parser.add_argument('--forget_rate', type=float, default=None)
    parser.add_argument('--noise_type', type=str, default='pairflip')
    parser.add_argument('--n_epoch', type=int, default= 100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_iter_per_epoch', type=int, default=400)
    parser.add_argument('--num_gradual', type=int, default=10, help='Number of gradual epochs')
    parser.add_argument('--exponent', type=float, default=1.0, help='Exponent for the forget rate schedule') 
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Define Forget Rate (if not explicitly set)
    forget_rate = args.forget_rate or args.noise_rate

    # Define Drop Rate Schedule
    global rate_schedule
    rate_schedule = np.ones(args.n_epoch) * forget_rate
    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)


    # Dataset Paths
    train_csv = "/projects/0/prjs1245/Co-teaching/train.csv"
    test_csv = "/projects/0/prjs1245/Co-teaching/test_total.csv"
    root_dir = "/projects/0/prjs0830/DATA/ADNI/Imaging/Preprocessed"
    

    # Transforms
    transform_train = Compose([
        LoadImage(ensure_channel_first=True),
        Resize(spatial_size=(96, 116, 96)),
        NormalizeIntensity(nonzero=True, channel_wise=True), 
    ])
    
    transform_test = Compose([
        LoadImage(ensure_channel_first=True),
        Resize(spatial_size=(96, 116, 96)),
        NormalizeIntensity(nonzero=True, channel_wise=True),  
    ])

    # Load Datasets
    print("creating ADNI dataset")
    train_dataset = ADNI(csv_path=train_csv, root=root_dir, transform=transform_train, train=True, noise_type=args.noise_type, noise_rate=args.noise_rate)
    test_dataset = ADNI(csv_path=test_csv, root=root_dir, transform=transform_test, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=args.num_workers, drop_last= True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers)

    noise_or_not = torch.Tensor(train_dataset.noise_or_not).cuda()
    
    # Model Initialization for adni
    cnn1 = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).cuda()
    cnn2 = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).cuda()
    
    model1=cnn1
    model2=cnn2

    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=args.lr)

    losses_1 = []
    losses_2 = []
    accs_1 = []
    accs_2 = []
    accuracy_toget = []
    # Training Loop

    for epoch in range(args.n_epoch):
        cnn1.train()
        cnn2.train()

    
        mean_pure_ratio1, mean_pure_ratio2, loss1_list, loss2_list, index_list, ep_loss1, ep_loss2, discarded_samples = train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        acc1, acc2 = evaluate(test_loader, cnn1, cnn2)

        _, _, acc_combined = evaluatefinal(test_loader, cnn1, cnn2)

        losses_1.append(ep_loss1)
        losses_2.append(ep_loss2)

        accs_1.append(acc1)
        accs_2.append(acc2)
        accuracy_toget.append(acc_combined)
        

        print(f"Epoch [{epoch+1}/{args.n_epoch}], Loss Model 1: {ep_loss1}, Loss Model 2: {ep_loss2}, Test Accuracy: Model1 {acc1:.2f}%, Model2 {acc2:.2f}%, Pure Ratio1: {mean_pure_ratio1:.2f}%, Pure Ratio2: {mean_pure_ratio2:.2f}%, Final Accucracy {acc_combined:.2f}% ")
        
        if epoch == args.n_epoch - 1:  # Save CSV only after the last epoch
            final_loss_data = []
            
            #voor ADNI
            for idx, ptid in enumerate(train_loader.dataset.data):  # Loop over dataset
                original_label = train_loader.dataset.labels[idx]  # Original label before noise
                noisy_label = train_loader.dataset.noisy_labels[idx]  # Noisy label after noise
                noise_status = train_loader.dataset.noise_or_not[idx]  # 1 = noisy, 0 = clean
                loss_model1 = loss1_list[idx] if idx < len(loss1_list) else None  # Loss per sample from last epoch
                loss_model2 = loss2_list[idx] if idx < len(loss2_list) else None  # Loss per sample from last epoch

                final_loss_data.append({
                    "PTID": ptid,
                    "Original_Label": original_label,
                    "Noisy_Label": noisy_label,
                    "Noise_Status": noise_status,
                    "Loss_Model1": loss_model1,
                    "Loss_Model2": loss_model2
                })
           
            

    # Save ALL discarded samples at the end of training**
    df_discarded = pd.DataFrame(discarded_samples)
    discarded_csv_path = "resultsCT/discarded_samples_last_epoch_alz_noisetype.csv"
    df_discarded.to_csv(discarded_csv_path, index=False)

    print(f"Discarded samples for all epochs saved to {discarded_csv_path}")

    # Convert to DataFrame
    df_final_loss = pd.DataFrame(final_loss_data)

    # Save to CSV
    final_csv_path = "resultsCT/final_noise_labels_and_loss_alz_age_noisetypecsv"
    df_final_loss.to_csv(final_csv_path, index=False)
    
    print(f"Final loss and noise status CSV saved: {final_csv_path}")

    epochs = list(range(args.n_epoch))  
    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot training loss on the primary y-axis
    ax1.plot(epochs, losses_1, color='blue', marker='o', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss Model 1', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for validation accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy_toget, color='green', marker='x', label='Validation Accuracy')
    ax2.set_ylabel('Validation Accuracy (%) Model 1', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title
    plt.title('Training Loss and Validation Accuracy per Epoch')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the figure as PNG
    plt.tight_layout()
    plt.savefig('training_loss_validation_accuracy_model1_alz_noisetype.png', dpi=300)  

    # Show the plot (optional)
    plt.show()    

    # Create a figure and axis for Model 2
    fig, ax1 = plt.subplots()

    # Plot training loss on the primary y-axis
    ax1.plot(epochs, losses_2, color='orange', marker='o', label='Training Loss Model 2')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss Model 2', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a secondary y-axis for validation accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy_toget, color='red', marker='x', label='Validation Accuracy Model 2')
    ax2.set_ylabel('Validation Accuracy (%) Model 2', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title
    plt.title('Training Loss and Validation Accuracy per Epoch - Model 2')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the figure as PNG
    plt.tight_layout()
    plt.savefig('training_loss_validation_accuracy_model2_alz_noisetype.png', dpi=300) 

    # Show the plot 
    plt.show()
   
    # Final prediction probabilities (averaged logits from both models)

    final_probs, final_labels, acc_combined = evaluatefinal(test_loader, cnn1, cnn2)

    # Save as CSV
    column_names = [f"Prob_Class_{i}" for i in range(final_probs.shape[1])]
    df_final_probs = pd.DataFrame(final_probs, columns=column_names)
    df_final_probs["True_Label"] = final_labels
    probs_csv_path = "resultsCT/final_probs_and_labels_alz_age_noisetype.csv"
    df_final_probs.to_csv(probs_csv_path, index=False)

    print(f"Saved final predicted probabilities to: {probs_csv_path}")


