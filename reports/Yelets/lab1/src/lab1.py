import numpy as np

c0 = 7
c1 = -7
LR = 0.5
EE = 0.01
MAX_EPOCHS = 10000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1.0 - y)

def normalize(value):
    return (value - c0) / (c1 - c0)

def denormalize(value):
    return c0 + value * (c1 - c0)

def get_class(value):
    if abs(value - c0) <= abs(value - c1):
        return c0
    return c1

data = np.array([
    [c0, c0, c0],
    [c0, c1, c1],
    [c1, c0, c1],
    [c1, c1, c0]
], dtype=float)

X = data[:, :2]
Y = data[:, 2]

Xn = normalize(X)
Yn = normalize(Y)

class MLP:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.b1 = rng.uniform(-0.1, 0.1, 2)
        self.W2 = rng.uniform(-0.1, 0.1, 2)
        self.b2 = float(rng.uniform(-0.1, 0.1))

    def forward(self, x):
        S1 = np.dot(x, self.W1) + self.b1
        h = sigmoid(S1)
        S2 = np.dot(h, self.W2) + self.b2
        y = sigmoid(S2)
        return S1, h, S2, y

    def train(self, X, Y):
        epoch = 0

        while epoch < MAX_EPOCHS:
            for x, target in zip(X, Y):
                S1, h, S2, y = self.forward(x)
                delta2 = (y - target) * sigmoid_derivative(y)
                old_W2 = self.W2.copy()
                self.W2 -= LR * h * delta2
                self.b2 -= LR * delta2
                delta1 = old_W2 * delta2 * sigmoid_derivative(h)
                self.W1 -= LR * np.outer(x, delta1)
                self.b1 -= LR * delta1

            epoch += 1
            Es = self.calculate_error(X, Y)
            if Es <= EE:
                return epoch, Es, True

        Es = self.calculate_error(X, Y)
        return epoch, Es, False

    def calculate_error(self, X, Y):
        error = 0.0
        for x, target in zip(X, Y):
            _, _, _, y = self.forward(x)
            error += 0.5 * (y - target) ** 2
        return error

    def predict(self, x):
        _, _, _, y = self.forward(normalize(x))
        return float(denormalize(y))

class SLP:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.1, 0.1, 2)
        self.b = float(rng.uniform(-0.1, 0.1))

    def forward(self, x):
        S = np.dot(x, self.W) + self.b
        y = sigmoid(S)
        return S, y

    def train(self, X, Y):
        epoch = 0

        while epoch < MAX_EPOCHS:
            for x, target in zip(X, Y):
                S, y = self.forward(x)
                delta = (y - target) * sigmoid_derivative(y)
                self.W -= LR * x * delta
                self.b -= LR * delta

            epoch += 1
            Es = self.calculate_error(X, Y)
            if Es <= EE:
                return epoch, Es, True

        Es = self.calculate_error(X, Y)
        return epoch, Es, False

    def calculate_error(self, X, Y):
        error = 0.0

        for x, target in zip(X, Y):
            _, y = self.forward(x)
            error += 0.5 * (y - target) ** 2
        return error

    def predict(self, x):
        _, y = self.forward(normalize(x))
        return float(denormalize(y))

def calculate_accuracy(network, X, Y):
    correct = 0

    for x, target in zip(X, Y):
        predicted = network.predict(x)
        if get_class(predicted) == target:
            correct += 1
    return correct / len(Y) * 100

def print_results(network, name):
    print(name)
    print(f"{'A':>8} {'B':>8} {'Ожидаемое':>10} {'y':>12} {'Класс':>10}")

    for x, target in zip(X, Y):
        predicted = network.predict(x)
        predicted_class = get_class(predicted)
        print(
            f"{x[0]:8.2f} "
            f"{x[1]:8.2f} "
            f"{target:10.2f} "
            f"{predicted:12.6f} "
            f"{predicted_class:10.2f}"
        )

def main():
    mlp = MLP()
    mlp_epochs, mlp_error, _ = mlp.train(Xn, Yn)
    mlp_accuracy = calculate_accuracy(mlp, X, Y)

    slp = SLP()
    slp_epochs, slp_error, _ = slp.train(Xn, Yn)
    slp_accuracy = calculate_accuracy(slp, X, Y)

    print("\nMLP")
    print(f"Количество эпох: {mlp_epochs}")
    print(f"Ошибка: {mlp_error:.8f}")
    print(f"Точность: {mlp_accuracy:.2f}%")

    print("\nSLP")
    print(f"Количество эпох: {slp_epochs}")
    print(f"Ошибка: {slp_error:.8f}")
    print(f"Точность: {slp_accuracy:.2f}%")

    print_results(mlp, "\nMLP результаты:")
    print_results(slp, "\nSLP результаты:")
    print("\nВведите A и B из диапазона [-10; 10], для выхода введите q")

    while True:
            a = input("A = ")
            if a.lower() == "q":
                break

            b = input("B = ")
            if b.lower() == "q":
                break

            a = float(a)
            b = float(b)
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                print("Значения должны находиться в диапазоне [-10; 10]!")
                continue

            result = mlp.predict(np.array([a, b]))
            predicted_class = get_class(result)
            print(f"y = {result:.6f}")
            print(f"Класс: {predicted_class}")

if __name__ == "__main__":
    main()