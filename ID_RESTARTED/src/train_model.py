import os, math, random 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from siamese_model import cosine_sim, contrastive_loss, SimeseNet
from dataset_pairer import PairDataset
from torch.utils.data import DataLoader
import torch
import os
import math
import random

positive_images_folder = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\positives"

negative_images_folder = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\celebA_negative"

output_folder = r"D:\python\ID CARD DETECTION\ID_RESTARTED\data\siamese"
checkpoint = os.path.join(output_folder, "siamese.pth")

batch_size = 32
epochs = 15
learning_rate = 1e-4
img_size = 224
pairs_per_epoch = 4000
device = "cuda" if torch.cuda.is_available() else "cpu"
margin = 1.0
threshold_output = os.path.join(output_folder, "best_threshold.txt")

os.makedirs(output_folder, exist_ok=True)
torch.manual_seed(42)
random.seed(42)


def evaluation_set_builder(positive_images_folder, negative_images_folder, num_of_positive=100, num_of_negative=100, img_size=224):
    dataset_eval = PairDataset(
        positive_images_folder, negative_images_folder, pairs_per_epoch=num_of_negative+num_of_positive, img_size=img_size)
    return dataset_eval


def best_threshold_finder(sims, labels):
    best_threshold, best_f1 = 0.0, -1.0
    for t in np.linspace(-1.0, 1.0, 201):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1


def main():
    dataset = PairDataset(positive_images_folder, negative_images_folder,
                          pairs_per_epoch=pairs_per_epoch, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=(device == 'cuda'))
    model = SimeseNet(embed_dim=128).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset_eval = evaluation_set_builder(
        positive_images_folder, negative_images_folder, num_of_positive=100, num_of_negative=100, img_size=img_size)
    dataloader_eval = DataLoader(
        dataset_eval, batch_size=64, shuffle=False, num_workers=2, pin_memory=(device == 'cuda'))
    best_accuracy = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for x1, x2, y, in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            z1, z2 = model(x1, x2)
            loss = contrastive_loss(z1, z2, y, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()*x1.size(0)

        training_loss = running/len(dataset)
        print(
            f"Epoch number- {epoch} out of {epochs} and the training loss is: {training_loss:.4f}")

        model.eval()
        all_sims, all_y = [], []
        with torch.no_grad():
            for x1, x2, y in dataloader_eval:
                x1 = x1.to(device)
                x2 = x2.to(device)
                z1, z2 = model(x1, x2)
                similarity = cosine_sim(z1, z2).cpu().numpy()
                all_sims.extend(similarity.tolist())
                all_y.extend(y.numpy().tolist())
        try:
            area_under_curve = roc_auc_score(all_y, all_sims)
        except:
            area_under_curve = float("nan")
        threshold, f1 = best_threshold_finder(
            sims=torch.tensor(all_sims).numpy(),
            labels=torch.tensor(all_y).numpy()
        )
        # Calculate accuracy with the best threshold
        preds = (torch.tensor(all_sims).numpy() >= threshold).astype(int)
        accuracy = accuracy_score(torch.tensor(all_y).numpy(), preds)

        print(f"Evaluation: \n Accuracy= {accuracy:.3f} \n Area under the curve= {area_under_curve:.3f} \n Best threshold= {threshold:.3f} \n F1 score= {f1:.3f}")

        improved = accuracy > best_accuracy
        if improved:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint)
            with open(threshold_output, "w") as f:
                f.write(str(threshold))
            print(
                f"Saved checkpoint:{checkpoint} \n Threshold: {threshold:.3f} \n Accuracy: {accuracy:.3f}")

if __name__ =="__main__":
    main()