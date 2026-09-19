import numpy as np


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)


class MinMaxScaler:
    def __init__(self, c0, c1):
        self.c0 = c0
        self.c1 = c1
        self.v_min = min(c0, c1)
        self.v_max = max(c0, c1)
        self.span = self.v_max - self.v_min

    def norm(self, v):
        return (np.asarray(v, dtype=float) - self.v_min) / self.span

    def denorm(self, v):
        return self.v_min + np.asarray(v, dtype=float) * self.span

    def nearest_class(self, v_real):
        if abs(v_real - self.c0) <= abs(v_real - self.c1):
            return self.c0
        return self.c1


class MLP221:
    def __init__(self, lr=2.0, w_init=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.uniform(-w_init, w_init, size=(2, 2))
        self.T1 = rng.uniform(-w_init, w_init, size=(2,))
        self.W2 = rng.uniform(-w_init, w_init, size=(2,))
        self.T2 = rng.uniform(-w_init, w_init, size=(1,))[0]

    def forward(self, x):
        s_hidden = x @ self.W1 + self.T1
        y_hidden = sigmoid(s_hidden)
        s_out = y_hidden @ self.W2 + self.T2
        y_out = sigmoid(s_out)
        return y_hidden, y_out

    def train_step(self, x, e):
        y_hidden, y_out = self.forward(x)

        delta_out = y_out - e
        g_out = delta_out * sigmoid_derivative_from_output(y_out)

        delta_hidden = g_out * self.W2
        g_hidden = delta_hidden * sigmoid_derivative_from_output(y_hidden)

        self.W2 -= self.lr * g_out * y_hidden
        self.T2 -= self.lr * g_out
        self.W1 -= self.lr * np.outer(x, g_hidden)
        self.T1 -= self.lr * g_hidden

        return y_out

    def predict(self, x):
        _, y_out = self.forward(np.asarray(x, dtype=float))
        return y_out

    def total_error(self, X, E):
        es = 0.0
        for x, e in zip(X, E):
            _, y_out = self.forward(x)
            es += 0.5 * (y_out - e) ** 2
        return es

    def fit(self, X, E, eps=0.01, max_epochs=20000, rng=None, shuffle=True):
        rng = rng or np.random.default_rng()
        order = np.arange(len(X))
        for epoch in range(1, max_epochs + 1):
            if shuffle:
                rng.shuffle(order)
            for idx in order:
                self.train_step(X[idx], E[idx])
            es = self.total_error(X, E)
            if es <= eps:
                return epoch, es
        return max_epochs, es


class SingleLayerSigmoidPerceptron:
    def __init__(self, lr=2.0, w_init=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = rng.uniform(-w_init, w_init, size=(2,))
        self.T = rng.uniform(-w_init, w_init, size=(1,))[0]

    def forward(self, x):
        s = x @ self.W + self.T
        return sigmoid(s)

    def train_step(self, x, e):
        y = self.forward(x)
        delta = y - e
        g = delta * sigmoid_derivative_from_output(y)
        self.W -= self.lr * g * x
        self.T -= self.lr * g
        return y

    def predict(self, x):
        return self.forward(np.asarray(x, dtype=float))

    def total_error(self, X, E):
        es = 0.0
        for x, e in zip(X, E):
            y = self.forward(x)
            es += 0.5 * (y - e) ** 2
        return es

    def fit(self, X, E, eps=0.01, max_epochs=20000, rng=None, shuffle=True):
        rng = rng or np.random.default_rng()
        order = np.arange(len(X))
        for epoch in range(1, max_epochs + 1):
            if shuffle:
                rng.shuffle(order)
            for idx in order:
                self.train_step(X[idx], E[idx])
            es = self.total_error(X, E)
            if es <= eps:
                return epoch, es
        return max_epochs, es


def accuracy(model, X, E_raw, scaler):
    correct = 0
    rows = []
    for x, e_true in zip(X, E_raw):
        y_norm = model.predict(x)
        y_real = float(scaler.denorm(y_norm))
        pred_class = scaler.nearest_class(y_real)
        ok = np.isclose(pred_class, e_true)
        correct += int(ok)
        rows.append((x, y_real, pred_class, e_true, ok))
    return correct / len(X), rows


def main():
    c0, c1 = 5, -2
    scaler = MinMaxScaler(c0, c1)

    raw_table = [
        (5, 5, 5),
        (5, -2, -2),
        (-2, 5, -2),
        (-2, -2, 5),
    ]
    X_raw = np.array([[a, b] for a, b, _ in raw_table], dtype=float)
    E_raw = np.array([t for _, _, t in raw_table], dtype=float)
    X = scaler.norm(X_raw)
    E = scaler.norm(E_raw)

    EPS = 0.01
    MAX_EPOCHS = 20000
    LR = 2.0
    SEED = 1

    mlp = MLP221(lr=LR, seed=SEED)
    mlp_epochs, mlp_err = mlp.fit(X, E, eps=EPS, max_epochs=MAX_EPOCHS, rng=np.random.default_rng(SEED))
    mlp_acc, mlp_rows = accuracy(mlp, X, E_raw, scaler)

    slp = SingleLayerSigmoidPerceptron(lr=LR, seed=SEED)
    slp_epochs, slp_err = slp.fit(X, E, eps=EPS, max_epochs=MAX_EPOCHS, rng=np.random.default_rng(SEED))
    slp_acc, slp_rows = accuracy(slp, X, E_raw, scaler)

    print(f"MLP 2-2-1: эпох={mlp_epochs}, E={mlp_err:.6f}, точность={mlp_acc:.0%}")
    for (a, b), (x, y_real, pred_class, e_true, ok) in zip(X_raw, mlp_rows):
        print(f"  ({a:.0f},{b:.0f}) -> {y_real:.4f} ({pred_class:.0f}), ожид. {e_true:.0f}")

    print()
    print(f"Перцептрон 2-1: эпох={slp_epochs}, E={slp_err:.6f}, точность={slp_acc:.0%}")
    for (a, b), (x, y_real, pred_class, e_true, ok) in zip(X_raw, slp_rows):
        print(f"  ({a:.0f},{b:.0f}) -> {y_real:.4f} ({pred_class:.0f}), ожид. {e_true:.0f}")

    print("\n--- Режим ввода (выход: q) ---")
    while True:
        raw = input("A B [-10..10]: ").strip()
        if not raw or raw.lower() == "q":
            break
        try:
            a_str, b_str = raw.replace(",", " ").split()
            a, b = float(a_str), float(b_str)
        except ValueError:
            print("  Некорректный ввод")
            continue
        x = scaler.norm(np.array([a, b], dtype=float))
        y_real = float(scaler.denorm(mlp.predict(x)))
        cls = scaler.nearest_class(y_real)
        print(f"  y={y_real:.4f}, ближе к c={cls:.0f}")


if __name__ == "__main__":
    main()