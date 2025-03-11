import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Load Data
# -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

obs = torch.load('obs.pt', weights_only=True).to(device)          # (B, 17)
labels = torch.load('phase.pt', weights_only=True).int().to(device)     # (B,)



# -----------------------------
# 2. Create Custom Dataset
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

full_dataset = CustomDataset(obs, labels)

# -----------------------------
# 3. Train / Val / Test Split
# -----------------------------
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

# -----------------------------
# 4. Define Model
# -----------------------------
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = FeedForwardNN().to(device)

# -----------------------------
# 5. Loss and Optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6. Training Loop with Validation
# -----------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x).squeeze()
            loss = criterion(outputs, y.float())
            preds = (outputs >= 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Train
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set after each epoch
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# -----------------------------
# 7. Final Evaluation on Test Set
# -----------------------------
test_loss, test_acc = evaluate(test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

torch.save(model.state_dict(), "./model.pth")
