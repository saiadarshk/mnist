import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------- Hyperparameters --------
TRAIN_SIZE = 10000
epochs = 10
learning_rate = 1e-2
batch_size = 32

torch.set_float32_matmul_precision("high")

# -------- Load Data --------
X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test_np = np.fromfile("data/y_test.bin", dtype=np.int32)

# -------- Normalize --------
mean, std = 0.1307, 0.3081
X_train_np = (X_train_np - mean) / std
X_test_np = (X_test_np - mean) / std

# -------- Convert to Tensors --------
train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(-1, 1, 28, 28)).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda")
test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda")
test_labels = torch.from_numpy(y_test_np).long().to("cuda")

iters_per_epoch = TRAIN_SIZE // batch_size

# -------- Model --------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # FIXED
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 1024, 10).to("cuda")

# -------- He Initialization --------
with torch.no_grad():
    fan_in = model.fc1.weight.size(1)
    scale = (2.0 / fan_in) ** 0.5
    model.fc1.weight.uniform_(-scale, scale)
    model.fc1.bias.zero_()

    fan_in = model.fc2.weight.size(1)
    scale = (2.0 / fan_in) ** 0.5
    model.fc2.weight.uniform_(-scale, scale)
    model.fc2.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# -------- Accuracy --------
def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct, total

# -------- Training --------
def train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses, epoch_accuracies):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(iters_per_epoch):

        # Data loading timing
        data_start = time.time()
        data = train_data[i * batch_size:(i + 1) * batch_size]
        target = train_labels[i * batch_size:(i + 1) * batch_size]
        data_end = time.time()
        timing_stats['data_loading'] += data_end - data_start

        optimizer.zero_grad()

        # Forward timing
        forward_start = time.time()
        outputs = model(data)
        forward_end = time.time()
        timing_stats['forward'] += forward_end - forward_start

        # Loss timing
        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation'] += loss_end - loss_start

        # Accuracy
        batch_correct, batch_total = compute_accuracy(outputs, target)
        correct += batch_correct
        total += batch_total

        # Backward timing
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start

        # Update timing
        update_start = time.time()
        optimizer.step()
        update_end = time.time()
        timing_stats['weight_updates'] += update_end - update_start

    epoch_losses.append(epoch_loss / iters_per_epoch)
    epoch_accuracy = 100 * correct / total
    epoch_accuracies.append(epoch_accuracy)

# -------- Evaluation --------
def evaluate(model, test_data, test_labels):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(test_data) // batch_size):
            data = test_data[i * batch_size:(i + 1) * batch_size]
            target = test_labels[i * batch_size:(i + 1) * batch_size]

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

# -------- Main --------
if __name__ == "__main__":
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }

    epoch_losses = []
    epoch_accuracies = []

    total_start = time.time()

    for epoch in range(epochs):
        train_timed(model, criterion, optimizer, epoch,
                    timing_stats, epoch_losses, epoch_accuracies)

        print(f"Epoch {epoch} loss: {epoch_losses[epoch]:.4f}, accuracy: {epoch_accuracies[epoch]:.2f}%")

    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start

    print("\n=== PYTORCH CUDA IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")

    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")

    print("Finished Training")

    # -------- Final Accuracy --------
    evaluate(model, test_data, test_labels)
