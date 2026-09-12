# local files
from synthetic_dataset import SyntheticEllipseDataset
from resunet import ResUNet
import utils

# libraries
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device.type)

dataset = SyntheticEllipseDataset(num_samples=128)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

print('Visualizando uma amostra do dataset...')
image, mask, instance_map = train_dataset[0]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

axes[0].imshow(image.squeeze().permute(1, 2, 0), cmap="gray")
axes[0].set_title("Image")
axes[0].axis("off")

axes[1].imshow(mask.squeeze(), cmap="gray")
axes[1].set_title("Binary mask")
axes[1].axis("off")

axes[2].imshow(instance_map.squeeze(), cmap="inferno")
axes[2].set_title("Instance map")
axes[2].axis("off")

plt.tight_layout()
plt.show()


# Treinando a ResUNet no dataset sintético ----------------------------------

print('Treinando a ResUNet no dataset...')
epoches = 1
model = ResUNet(3, 1).to(device)

loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(epoches):
    for batch, (image, mask, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch")):
        image, mask = image.to(device), mask.to(device)

        y_hat = model(image).squeeze(1)
        loss = loss_fn(y_hat, mask)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Verificando a saída do modelo ------------------------------------------

model.eval()

for img, mask, _ in DataLoader(test_dataset, shuffle=True):
    img, mask = img.to(device), mask.to(device)

    y = F.sigmoid(model(img))[0, 0].cpu().detach().numpy()
    img = img[0].permute(1, 2, 0).cpu().detach().numpy()
    mask = mask[0].cpu().detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Answer")
    axes[1].axis("off")

    axes[2].imshow(y, cmap="gray")
    axes[2].set_title("Model answer")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    break


# computa o IoU e Dice no conjunto de validação/teste ------------------------------------------

def pred(image):
    return torch.sigmoid(model(image.to(device)))[0].cpu()


thresholds = torch.arange(0, 1, 0.05)

with torch.no_grad():
    # encontra o melhor limiar com o conjunto de validação
    iou_scores, dice_scores = [], []
    for image, mask, _ in DataLoader(val_dataset):
        y_hat = pred(image)
        iou_scores.append([float(utils.iou_score(1 * (y_hat > th), mask)) for th in thresholds])
        dice_scores.append([float(utils.dice_score(1 * (y_hat > th), mask)) for th in thresholds])

    iou_scores = torch.tensor(iou_scores).mean(dim=0)
    dice_scores = torch.tensor(dice_scores).mean(dim=0)
    best_threshold = thresholds[iou_scores.argmax()]

    plt.axvline(best_threshold, color="k", linestyle="--")
    plt.plot(thresholds, iou_scores, label="IoU")
    plt.plot(thresholds, dice_scores, label="Dice")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # computa o IoU e Dice pontuais no conjunto de teste
    iou_scores_test, dice_scores_test = [], []
    for image, mask, _ in DataLoader(test_dataset, pin_memory=True):
        y_hat = pred(image)
        iou_scores_test.append(float(utils.iou_score(1 * (y_hat > best_threshold), mask)))
        dice_scores_test.append(float(utils.dice_score(1 * (y_hat > best_threshold), mask)))

    final_iou_score = torch.tensor(iou_scores_test).mean().item()
    final_dice_score = torch.tensor(dice_scores_test).mean().item()

    print("IoU final:", final_iou_score)
    print("Dice final:", final_dice_score)
