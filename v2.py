import numpy as np
import time

# -------- Load Data --------
X_train = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)[:10000]
y_train = np.fromfile("data/y_train.bin", dtype=np.int32)[:10000]
X_test = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test = np.fromfile("data/y_test.bin", dtype=np.int32)

# -------- Normalize --------
mean, std = 0.1307, 0.3081
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# -------- Reshape --------
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# -------- Activations --------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# -------- Layers --------
def initialize_weights(input_size, output_size):
    scale = np.sqrt(2.0 / input_size)
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def initialize_bias(output_size):
    return np.zeros((1, output_size))

def linear_forward(x, weights, bias):
    return x @ weights + bias

def linear_backward(grad_output, x, weights):
    grad_weights = x.T @ grad_output
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)
    grad_input = grad_output @ weights.T
    return grad_input, grad_weights, grad_bias

# -------- Softmax + Loss --------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    probs = softmax(y_pred)
    log_probs = np.log(probs[np.arange(batch_size), y_true])
    return -np.sum(log_probs) / batch_size

# -------- Accuracy --------
def compute_accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=1)
    correct = np.sum(preds == y_true)
    return correct, len(y_true)

# -------- Model --------
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = initialize_weights(input_size, hidden_size)
        self.b1 = initialize_bias(hidden_size)
        self.w2 = initialize_weights(hidden_size, output_size)
        self.b2 = initialize_bias(output_size)

    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.reshape(bs, -1)
        z1 = linear_forward(x_flat, self.w1, self.b1)
        a1 = relu(z1)
        z2 = linear_forward(a1, self.w2, self.b2)
        return z2, (x_flat, z1, a1)

    def backward(self, grad_output, cache):
        x, z1, a1 = cache

        dz2, dw2, db2 = linear_backward(grad_output, a1, self.w2)
        dz1 = dz2 * relu_derivative(z1)
        dx, dw1, db1 = linear_backward(dz1, x, self.w1)

        return dw1, db1, dw2, db2

    def update(self, dw1, db1, dw2, db2, lr):
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

# -------- Evaluation --------
def evaluate(model, X_test, y_test, batch_size):
    correct = 0
    total = 0

    for i in range(0, len(X_test), batch_size):
        Xb = X_test[i:i+batch_size]
        yb = y_test[i:i+batch_size]

        y_pred, _ = model.forward(Xb)
        preds = np.argmax(y_pred, axis=1)

        correct += np.sum(preds == yb)
        total += len(yb)

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# -------- Training --------
def train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, lr):

    timing = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }

    total_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(X_train), batch_size):

            # Data loading
            t0 = time.time()
            Xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            t1 = time.time()
            timing['data_loading'] += t1 - t0

            # Forward
            t0 = time.time()
            y_pred, cache = model.forward(Xb)
            t1 = time.time()
            timing['forward'] += t1 - t0

            # Loss
            t0 = time.time()
            loss = cross_entropy_loss(y_pred, yb)
            epoch_loss += loss

            # Accuracy
            c, t = compute_accuracy(y_pred, yb)
            correct += c
            total += t

            probs = softmax(y_pred)
            one_hot = np.zeros_like(y_pred)
            one_hot[np.arange(len(yb)), yb] = 1
            grad = (probs - one_hot) / len(yb)
            t1 = time.time()
            timing['loss_computation'] += t1 - t0

            # Backward
            t0 = time.time()
            dw1, db1, dw2, db2 = model.backward(grad, cache)
            t1 = time.time()
            timing['backward'] += t1 - t0

            # Update
            t0 = time.time()
            model.update(dw1, db1, dw2, db2, lr)
            t1 = time.time()
            timing['weight_updates'] += t1 - t0

        print(f"Epoch {epoch} loss: {epoch_loss / (len(X_train)//batch_size):.4f}, accuracy: {100*correct/total:.2f}%")

    timing['total_time'] = time.time() - total_start

    print("\n=== NUMPY IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing['total_time']:.1f} seconds\n")

    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing['data_loading']:6.3f}s ({100.0 * timing['data_loading']/timing['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing['forward']:6.3f}s ({100.0 * timing['forward']/timing['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing['loss_computation']:6.3f}s ({100.0 * timing['loss_computation']/timing['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing['backward']:6.3f}s ({100.0 * timing['backward']/timing['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing['weight_updates']:6.3f}s ({100.0 * timing['weight_updates']/timing['total_time']:5.1f}%)")

    print("Training completed!")

    evaluate(model, X_test, y_test, batch_size)

# -------- Main --------
if __name__ == "__main__":
    model = NeuralNetwork(784, 1024, 10)
    train_timed(model, X_train, y_train, X_test, y_test, 32, 10, 0.01)
