import numpy as np

C0, C1 = 3, -8
DATA = [(3, 3, 3), (3, -8, -8), (-8, 3, -8), (-8, -8, 3)]   # (A, B, A xor B)

#нормализация
T_LO, T_HI = 0.1, 0.9
Y_MIN, Y_MAX = min(C0, C1), max(C0, C1)

def norm_x(v):   return v / 10.0
def norm_y(y):   return T_LO + (y - Y_MIN) * (T_HI - T_LO) / (Y_MAX - Y_MIN)
def denorm_y(t): return Y_MIN + (t - T_LO) * (Y_MAX - Y_MIN) / (T_HI - T_LO)

X = np.array([[norm_x(a), norm_x(b)] for a, b, _ in DATA])
T = np.array([norm_y(y) for _, _, y in DATA])
Y_REAL = np.array([y for _, _, y in DATA], dtype=float)

sigmoid = lambda s: 1.0 / (1.0 + np.exp(-s))

def total_error(predict):
    return 0.5 * sum((T[i] - predict(X[i])) ** 2 for i in range(len(X)))

def accuracy(predict):
    ok = 0
    for i in range(len(X)):
        y = denorm_y(predict(X[i]))
        ok += abs(y - C0) < abs(y - C1) if Y_REAL[i] == C0 else abs(y - C1) < abs(y - C0)
    return ok / len(X)


class MLP:
    def __init__(self, rng):
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2)); self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2);      self.b2 = rng.uniform(-0.5, 0.5)

    def forward(self, x):
        self.h = sigmoid(x @ self.W1 + self.b1)
        self.y = sigmoid(self.h @ self.W2 + self.b2)
        return self.y

    def predict(self, x): return self.forward(x)

    def train_step(self, x, t, lr):
        y = self.forward(x)
        d_out = (y - t) * y * (1 - y)
        d_hid = d_out * self.W2 * self.h * (1 - self.h)
        self.W2 -= lr * d_out * self.h;  self.b2 -= lr * d_out
        self.W1 -= lr * np.outer(x, d_hid); self.b1 -= lr * d_hid


class SLP:
    def __init__(self, rng):
        self.w = rng.uniform(-0.5, 0.5, 2); self.b = rng.uniform(-0.5, 0.5)

    def predict(self, x): return sigmoid(x @ self.w + self.b)

    def train_step(self, x, t, lr):
        y = self.predict(x)
        d = (y - t) * y * (1 - y)
        self.w -= lr * d * x; self.b -= lr * d

def train(net, lr, Ee, max_epochs, rng):
    for epoch in range(1, max_epochs + 1):
        for i in rng.permutation(len(X)):
            net.train_step(X[i], T[i], lr)
        Es = total_error(net.predict)
        if Es <= Ee:
            return epoch, Es, True
    return max_epochs, Es, False

def report(name, net, epochs, Es, ok):
    print(f"\n=== {name} ===")
    print(f"Критерий остановки достигнут: {'да' if ok else 'НЕТ'}; эпох: {epochs}; Es = {Es:.6f}")
    print(f"{'A':>4} {'B':>4} {'цель':>5} {'выход':>8}  класс")
    for (a, b, y), x in zip(DATA, X):
        out = denorm_y(net.predict(x))
        cls = 'c0' if abs(out - C0) < abs(out - C1) else 'c1'
        print(f"{a:>4} {b:>4} {y:>5} {out:>8.3f}  {cls}")
    print(f"Accuracy: {accuracy(net.predict) * 100:.0f}%")

def run_mode(net):
    print(f"\nРежим функционирования (c0={C0}, c1={C1}). Пустая строка — выход.")
    while True:
        s = input("Введите A и B из [-10; 10] через пробел: ").strip()
        if not s: break
        try:
            a, b = map(float, s.split())
            assert -10 <= a <= 10 and -10 <= b <= 10
        except Exception:
            print("Некорректный ввод."); continue
        y = denorm_y(net.predict(np.array([norm_x(a), norm_x(b)])))
        cls = 'c0' if abs(y - C0) < abs(y - C1) else 'c1'
        print(f"ŷ = {y:.3f}  -> ближе к {cls} ({C0 if cls == 'c0' else C1})")

if __name__ == "__main__":
    LR, EE, MAX_EP = 0.5, 0.001, 200_000


    for seed in range(100):
        rng = np.random.default_rng(seed)
        mlp = MLP(rng)
        ep, Es, ok = train(mlp, LR, EE, MAX_EP, rng)
        if ok: break
        print(f"seed={seed}: не сошлась (Es={Es:.4f}), перезапуск")
    print(f"\n(seed = {seed})")
    report("Многослойный персептрон 2-2-1", mlp, ep, Es, ok)

    slp = SLP(np.random.default_rng(0))
    ep2, Es2, ok2 = train(slp, LR, EE, 20_000, np.random.default_rng(0))
    report("Однослойный персептрон", slp, ep2, Es2, ok2)

    run_mode(mlp)