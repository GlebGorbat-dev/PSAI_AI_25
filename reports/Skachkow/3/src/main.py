import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

C0, C1 = -3.0, -8.0
LR = 0.1
EE = 0.02
MAX_EPOCHS = 50000
SEEDS = [1, 2, 3, 4, 5]
INIT_SCALE = 1.0
B1_RANGE = (0.5, 1.0)

X_RAW = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]], dtype=float)
Y_RAW = np.array([C0, C1, C1, C0], dtype=float)  # A xor B


def to_norm(v):
    return (np.asarray(v, dtype=float) - C0) / (C1 - C0)


def from_norm(v):
    return C0 + np.asarray(v, dtype=float) * (C1 - C0)


X = to_norm(X_RAW)  # входы в {0, 1}
T = to_norm(Y_RAW)  # цели в {0, 1}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def sigmoid_d(a):
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0.0, z)


def relu_d(z):
    return (z > 0).astype(float)


def bce(y, t):
    eps = 1e-12
    y = np.clip(y, eps, 1 - eps)
    return -(t * np.log(y) + (1 - t) * np.log(1 - y))


class MLP:
    def __init__(self, hidden="sigmoid", seed=0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.W1 = rng.uniform(-INIT_SCALE, INIT_SCALE, (2, 2))
        self.b1 = rng.uniform(B1_RANGE[0], B1_RANGE[1], 2)
        self.W2 = rng.uniform(-INIT_SCALE, INIT_SCALE, 2)
        self.b2 = rng.uniform(-INIT_SCALE, INIT_SCALE)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = sigmoid(z1) if self.hidden == "sigmoid" else relu(z1)
        z2 = h @ self.W2 + self.b2
        y = sigmoid(z2)
        return z1, h, y

    def predict(self, x):
        return self.forward(np.asarray(x, dtype=float))[2]

    def train_step(self, x, t):
        z1, h, y = self.forward(x)
        err = bce(y, t)

        d2 = y - t
        if self.hidden == "sigmoid":
            d1 = d2 * self.W2 * sigmoid_d(h)
        else:
            d1 = d2 * self.W2 * relu_d(z1)
        self.W2 -= LR * d2 * h
        self.b2 -= LR * d2
        self.W1 -= LR * np.outer(x, d1)
        self.b1 -= LR * d1
        return err

    def fit(self, seed):
        rng = np.random.default_rng(1000 + seed)
        history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            Es = 0.0
            for i in rng.permutation(len(X)):  # онлайн-режим, случайный порядок
                Es += self.train_step(X[i], T[i])
            history.append(Es)
            if Es <= EE:
                return epoch, history, True
        return MAX_EPOCHS, history, False

    def evaluate(self):
        y = self.predict(X)
        Es = float(np.sum(bce(y, T)))
        acc = float(np.mean((y >= 0.5).astype(float) == T))
        mae = float(np.mean(np.abs(from_norm(y) - Y_RAW)))
        return Es, acc, mae

    def dead_neurons(self):
        """Скрытые нейроны ReLU, выдающие 0 на всех 4 обучающих примерах."""
        if self.hidden != "relu":
            return 0
        _, h, _ = self.forward(X)
        return int(np.sum(np.all(h <= 0, axis=0)))


def run_experiment():
    configs = {"A (Сигмоида)": "sigmoid", "B (ReLU)": "relu"}
    results = {}
    for name, act in configs.items():
        runs = []
        for s in SEEDS:
            net = MLP(act, seed=s)
            epochs, hist, ok = net.fit(s)
            Es, acc, mae = net.evaluate()
            runs.append(dict(seed=s, net=net, epochs=epochs, hist=hist, ok=ok,
                             Es=Es, acc=acc, mae=mae, dead=net.dead_neurons()))
        results[name] = runs
    return results


def print_tables(results):
    print(f"\nВариант 10: c0={C0:g}, c1={C1:g}; Ee={EE}; lr={LR}; max эпох={MAX_EPOCHS}\n")
    for name, runs in results.items():
        print(f"=== Конфигурация {name} ===")
        print(f"{'seed':>5} {'эпохи':>8} {'сошлась':>8} {'Es':>10} {'accuracy':>9} "
              f"{'MAE [c0;c1]':>12} {'мёртв.нейр.':>12}")
        for r in runs:
            print(f"{r['seed']:>5} {r['epochs']:>8} {str(r['ok']):>8} {r['Es']:>10.5f} "
                  f"{r['acc']:>9.2f} {r['mae']:>12.5f} {r['dead']:>12}")
        print()

    print("=== Сводная таблица (по 5 запускам) ===")
    print(f"{'Конфигурация':<16} {'сошлось':>8} {'эпохи mean':>11} {'эпохи std':>10} "
          f"{'min':>7} {'max':>7} {'Es mean':>10} {'acc mean':>9} {'MAE mean':>10}")
    for name, runs in results.items():
        ep = np.array([r["epochs"] for r in runs], dtype=float)
        print(f"{name:<16} {sum(r['ok'] for r in runs):>5}/{len(runs)} {ep.mean():>11.1f} "
              f"{ep.std():>10.1f} {int(ep.min()):>7} {int(ep.max()):>7} "
              f"{np.mean([r['Es'] for r in runs]):>10.5f} "
              f"{np.mean([r['acc'] for r in runs]):>9.2f} "
              f"{np.mean([r['mae'] for r in runs]):>10.5f}")
    print("(в 'эпохи' несошедшиеся запуски учтены значением MAX_EPOCHS)\n")


def representative(runs):
    ok = [r for r in runs if r["ok"]] or runs
    ok = sorted(ok, key=lambda r: r["epochs"])
    return ok[len(ok) // 2]


def plot_convergence(results):
    plt.figure(figsize=(8, 5))
    colors = {"A (Сигмоида)": "tab:blue", "B (ReLU)": "tab:red"}
    for name, runs in results.items():
        r = representative(runs)
        plt.plot(range(1, len(r["hist"]) + 1), r["hist"], color=colors[name],
                 label=f"{name}, seed={r['seed']}, {r['epochs']} эпох")
    plt.axhline(EE, color="gray", ls="--", label=f"Ee = {EE}")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es (BCE)")
    plt.title("Сходимость: Сигмоида vs ReLU (представительные запуски)")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_convergence.png", dpi=150)
    plt.close()


def plot_epochs_bar(results):
    n = len(SEEDS)
    x = np.arange(n)
    w = 0.38
    plt.figure(figsize=(8, 5))
    for k, (name, color) in enumerate([("A (Сигмоида)", "tab:blue"), ("B (ReLU)", "tab:red")]):
        ep = [r["epochs"] for r in results[name]]
        bars = plt.bar(x + (k - 0.5) * w, ep, w, color=color, label=name)
        for b, r in zip(bars, results[name]):
            if not r["ok"]:
                b.set_hatch("//")
                b.set_edgecolor("black")
    plt.xticks(x, [f"seed {s}" for s in SEEDS])
    plt.ylabel("Число эпох до Es ≤ Ee")
    plt.title("Разброс числа эпох по запускам (штриховка — не сошлось)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig2_epochs_bar.png", dpi=150)
    plt.close()


def plot_surfaces(results):
    grid = np.linspace(-10, 10, 300)
    AA, BB = np.meshgrid(grid, grid)
    pts = np.stack([to_norm(AA.ravel()), to_norm(BB.ravel())], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, runs) in zip(axes, results.items()):
        r = representative(runs)
        out = r["net"].predict(pts).reshape(AA.shape)
        im = ax.imshow(out, extent=[-10, 10, -10, 10], origin="lower",
                       cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.contour(AA, BB, out, levels=[0.5], colors="white", linewidths=1.5)
        for (a, b), t in zip(X_RAW, T):
            ax.scatter(a, b, c="red" if t == 1 else "orange", edgecolors="black",
                       s=90, zorder=5)
            ax.annotate(f"{from_norm(t):g}", (a, b), textcoords="offset points",
                        xytext=(6, 6), color="white", fontsize=9)
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_title(f"{name}, seed={r['seed']}")
        fig.colorbar(im, ax=ax, label="выход (норм. шкала)")
    plt.suptitle("Разделяющая поверхность в области [−10; 10]² (белая линия: y = 0.5)")
    plt.tight_layout()
    plt.savefig("fig3_surfaces.png", dpi=150)
    plt.close()


def classify(net, A, B):
    if not (-10 <= A <= 10 and -10 <= B <= 10):
        raise ValueError("A и B должны быть из [-10; 10]")
    y_norm = float(net.predict(to_norm([A, B])))
    y_raw = float(from_norm(y_norm))
    cls = 1 if y_norm >= 0.5 else 0
    return y_norm, y_raw, cls


def demo(results):
    extra = [(-5.5, -5.5), (-5.5, 0.0), (5.0, -8.0)]
    pairs = [tuple(p) for p in X_RAW] + extra
    for name, runs in results.items():
        r = representative(runs)
        print(f"--- Режим функционирования, конфигурация {name} (seed={r['seed']}) ---")
        print(f"{'A':>6} {'B':>6} | {'выход норм.':>11} {'выход [c0;c1]':>14} {'класс':>6}")
        for A, B in pairs:
            yn, yr, c = classify(r["net"], A, B)
            print(f"{A:>6g} {B:>6g} | {yn:>11.4f} {yr:>14.4f} {c:>6}")
        print()


def interactive(results):
    name = "B (ReLU)"
    print("Интерактивный режим. Выберите сеть: 1 — Сигмоида (A), 2 — ReLU (B)")
    name = "A (Сигмоида)" if input("> ").strip() == "1" else "B (ReLU)"
    net = representative(results[name])["net"]
    print(f"Сеть {name}. Введите A и B через пробел из [-10; 10], пустая строка — выход.")
    while True:
        s = input("A B > ").strip()
        if not s:
            break
        try:
            A, B = map(float, s.replace(",", ".").split())
            yn, yr, c = classify(net, A, B)
            print(f"  выход (норм.) = {yn:.4f}; выход [c0;c1] = {yr:.4f}; класс = {c}")
        except Exception as e:
            print("  ошибка:", e)


if __name__ == "__main__":
    results = run_experiment()
    print_tables(results)
    plot_convergence(results)
    plot_epochs_bar(results)
    plot_surfaces(results)
    print("Графики сохранены: fig1_convergence.png, fig2_epochs_bar.png, fig3_surfaces.png\n")
    demo(results)
    if "--interactive" in sys.argv:
        interactive(results)