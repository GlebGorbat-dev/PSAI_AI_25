import numpy as np

C0, C1 = 7, -7
X = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]], float)
Y = np.array([C0, C1, C1, C0], float)
TARGETS = (Y - C0) / (C1 - C0)
ERROR_LIMIT, MAX_EPOCHS, LEARNING_RATE = 0.005, 10000, 0.5


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -500, 500)))


def sigmoid_derivative(output):
    return output * (1 - output)


def normalize(values):
    return np.asarray(values, float) / 10


def to_real(output):
    return C0 + output * (C1 - C0)


def classify(value):
    return C1 if abs(value - C1) < abs(value - C0) else C0


class MLP:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = self.rng.uniform(-0.5, 0.5, 2)
        self.W2 = self.rng.uniform(-0.5, 0.5, (2, 1))
        self.b2 = self.rng.uniform(-0.5, 0.5, 1)

    def predict(self, values):
        x = normalize(values).ravel()
        self.hidden = sigmoid(x @ self.W1 + self.b1)
        return sigmoid(self.hidden @ self.W2 + self.b2).item()

    def train_step(self, values, target):
        output = self.predict(values)
        x = normalize(values).ravel()
        delta_output = (output - target) * output * (1 - output)
        delta_hidden = self.W2.ravel() * delta_output * sigmoid_derivative(self.hidden)
        self.W2 -= LEARNING_RATE * self.hidden.reshape(-1, 1) * delta_output
        self.b2 -= LEARNING_RATE * delta_output
        self.W1 -= LEARNING_RATE * np.outer(x, delta_hidden)
        self.b1 -= LEARNING_RATE * delta_hidden


class Perceptron:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.uniform(-0.5, 0.5, (2, 1))
        self.b = self.rng.uniform(-0.5, 0.5, 1)

    def predict(self, values):
        x = normalize(values).ravel()
        return sigmoid(x @ self.W + self.b).item()

    def train_step(self, values, target):
        output = self.predict(values)
        x = normalize(values).ravel()
        delta = (output - target) * output * (1 - output)
        self.W -= LEARNING_RATE * x.reshape(-1, 1) * delta
        self.b -= LEARNING_RATE * delta


def train(network):
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in network.rng.permutation(len(X)):
            network.train_step(X[i], TARGETS[i])
        error = 0.5 * sum((network.predict(x) - t) ** 2 for x, t in zip(X, TARGETS))
        if error <= ERROR_LIMIT:
            return epoch, error
    return MAX_EPOCHS, error


def print_results(name, network, epochs, error):
    outputs = [to_real(network.predict(x)) for x in X]
    accuracy = sum(classify(value) == target for value, target in zip(outputs, Y)) / len(Y)
    print(f"\n{name}: эпох={epochs}, E={error:.6f}, точность={accuracy * 100:.0f}%")
    for x, target, output in zip(X, Y, outputs):
        print(f"  ({x[0]:g},{x[1]:g}) -> {output:.4f} ({classify(output):g}), ожид. {target:g}")


def main():
    mlp, perceptron = MLP(), Perceptron()
    mlp_epochs, mlp_error = train(mlp)
    single_epochs, single_error = train(perceptron)
    print("c0=7, c1=-7")
    print_results("MLP 2-2-1", mlp, mlp_epochs, mlp_error)
    print_results("Перцептрон 2-1", perceptron, single_epochs, single_error)
    print("\nПроверочные примеры MLP:")
    for a, b in [(6, -6), (-6, 6), (0, 0)]:
        output = to_real(mlp.predict([a, b]))
        print(f"A={a:g}, B={b:g}: y={output:.4f}, класс={classify(output):g}")
    print("\nРежим ввода (выход: q)")
    while True:
        try:
            text = input("A B [-10..10]: ").strip()
        except EOFError:
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        try:
            a, b = map(float, text.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                raise ValueError
            output = to_real(mlp.predict([a, b]))
            print(f"y={output:.4f}, ближе к c={classify(output):g}")
        except ValueError:
            print("Введите два числа из [-10, 10]")


if __name__ == "__main__":
    main()
