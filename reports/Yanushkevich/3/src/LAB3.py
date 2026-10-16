import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)


def relu(s):
    return np.maximum(0.0, s)


def relu_derivative_from_input(s):
    return (s > 0).astype(float)


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
        return self.c0 if abs(v_real - self.c0) <= abs(v_real - self.c1) else self.c1


class MLP221:
    def __init__(self, lr=0.5, w_init=0.1, seed=None, hidden="sigmoid", bias_init=None):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.hidden = hidden
        self.W1 = rng.uniform(-w_init, w_init, size=(2, 2))
        if hidden == "relu" and bias_init is not None:
            self.T1 = rng.uniform(0.0, bias_init, size=(2,))
        else:
            self.T1 = rng.uniform(-w_init, w_init, size=(2,))
        self.W2 = rng.uniform(-w_init, w_init, size=(2,))
        self.T2 = rng.uniform(-w_init, w_init, size=(1,))[0]

    def forward(self, x):
        s_hidden = x @ self.W1 + self.T1
        if self.hidden == "relu":
            y_hidden = relu(s_hidden)
        else:
            y_hidden = sigmoid(s_hidden)
        s_out = y_hidden @ self.W2 + self.T2
        y_out = sigmoid(s_out)
        return s_hidden, y_hidden, y_out

    def train_step(self, x, e):
        s_hidden, y_hidden, y_out = self.forward(x)
        g_out = y_out - e

        delta_hidden = g_out * self.W2
        if self.hidden == "relu":
            g_hidden = delta_hidden * relu_derivative_from_input(s_hidden)
        else:
            g_hidden = delta_hidden * sigmoid_derivative_from_output(y_hidden)

        self.W2 -= self.lr * g_out * y_hidden
        self.T2 -= self.lr * g_out
        self.W1 -= self.lr * np.outer(x, g_hidden)
        self.T1 -= self.lr * g_hidden

    def predict(self, x):
        _, _, y_out = self.forward(np.asarray(x, dtype=float))
        return y_out

    def hidden_dead_fraction(self, X):
        dead = np.zeros(2)
        for x in X:
            s_hidden = x @ self.W1 + self.T1
            dead += (s_hidden <= 0).astype(float)
        return dead / len(X)

    def total_error(self, X, E):
        es = 0.0
        for x, e in zip(X, E):
            _, _, y = self.forward(x)
            yc = min(max(y, 1e-9), 1 - 1e-9)
            es += -(e * np.log(yc) + (1 - e) * np.log(1 - yc))
        return es

    def fit(self, X, E, eps, max_epochs, rng=None, shuffle=True):
        rng = rng or np.random.default_rng()
        order = np.arange(len(X))
        history = []
        for epoch in range(1, max_epochs + 1):
            if shuffle:
                rng.shuffle(order)
            for idx in order:
                self.train_step(X[idx], E[idx])
            es = self.total_error(X, E)
            history.append(es)
            if es <= eps:
                return epoch, es, history
        return max_epochs, es, history


def accuracy_and_mae(model, X, E_raw, scaler):
    correct, abs_err = 0, 0.0
    for x, e_true in zip(X, E_raw):
        y_real = float(scaler.denorm(model.predict(x)))
        pred_class = scaler.nearest_class(y_real)
        correct += int(np.isclose(pred_class, e_true))
        abs_err += abs(y_real - e_true)
    return correct / len(X), abs_err / len(X)


c0, c1 = 5, -2
scaler = MinMaxScaler(c0, c1)
raw_table = [(5, 5, 5), (5, -2, -2), (-2, 5, -2), (-2, -2, 5)]
X_raw = np.array([[a, b] for a, b, _ in raw_table], dtype=float)
E_raw = np.array([t for _, _, t in raw_table], dtype=float)
X = scaler.norm(X_raw)
E = scaler.norm(E_raw)

EPS = 0.05
MAX_EPOCHS = 20000
SEEDS = [1, 2, 3, 4, 5]

# Конфигурация А (Сигмоида) использует те же гиперпараметры, что и в ЛР №2.
# Для конфигурации В (ReLU) эмпирически подобраны отдельные lr и инициализация
# смещения T1 (положительный диапазон вместо симметричного) — без этого
# часть нейронов "мертва" уже при инициализации и не обучается ни при каком lr
# (см. вывод ниже и пояснение в выводах к лабораторной).
CONFIGS = {
    "A (Sigmoid)": dict(hidden="sigmoid", color="#2563eb", lr=0.5, bias_init=None),
    "B (ReLU)":    dict(hidden="relu",    color="#16a34a", lr=0.1, bias_init=0.1),
}

runs = {}
for name, cfg in CONFIGS.items():
    runs[name] = []
    for seed in SEEDS:
        m = MLP221(lr=cfg["lr"], seed=seed, hidden=cfg["hidden"], bias_init=cfg["bias_init"])
        ep, err, hist = m.fit(X, E, eps=EPS, max_epochs=MAX_EPOCHS, rng=np.random.default_rng(seed))
        converged = err <= EPS
        acc, mae = accuracy_and_mae(m, X, E_raw, scaler)
        dead = m.hidden_dead_fraction(X) if cfg["hidden"] == "relu" else None
        runs[name].append(dict(seed=seed, model=m, epochs=ep, err=err, hist=hist,
                                converged=converged, acc=acc, mae=mae, dead=dead))

print(f"{'Конфигурация':14s}{'seed':>6s}{'эпох':>8s}{'ошибка':>12s}{'accuracy':>10s}{'MAE':>10s}{'сошелся':>10s}{'мёртвые нейроны (доля)':>26s}")
for name, lst in runs.items():
    for r in lst:
        dead_str = f"{r['dead']}" if r["dead"] is not None else "-"
        print(f"{name:14s}{r['seed']:>6d}{r['epochs']:>8d}{r['err']:>12.5f}{r['acc']:>10.0%}{r['mae']:>10.3f}{str(r['converged']):>10s}{dead_str:>26s}")

print()
for name, lst in runs.items():
    n_conv = sum(r["converged"] for r in lst)
    ep_conv = [r["epochs"] for r in lst if r["converged"]]
    print(f"{name}: сошлось {n_conv}/5 запусков; эпохи (сошедшихся): {ep_conv if ep_conv else '-'}")

print(f"\nПримечание по конфигурации В (ReLU): lr={CONFIGS['B (ReLU)']['lr']}, "
      f"T1 инициализируется в [0; {CONFIGS['B (ReLU)']['bias_init']}] (а не симметрично около нуля),")
print("это подобрано специально, чтобы уменьшить число нейронов, которые『мертвы』уже при инициализации.")
print("Даже после этого у части seed остаётся нейрон с долей『мёртвых』состояний = 1.0 — это и есть")
print("граница применимости чистого подбора lr: если нейрон мёртв на всех 4 примерах обучающей выборки,")
print("его градиент тождественно равен нулю с первой эпохи и не восстанавливается ни при каком lr.")

rep = {
    "A (Sigmoid)": next((r for r in runs["A (Sigmoid)"] if r["converged"]), runs["A (Sigmoid)"][0]),
    "B (ReLU)": next((r for r in runs["B (ReLU)"] if r["converged"]), runs["B (ReLU)"][0]),
}

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, cfg in CONFIGS.items():
    r = rep[name]
    ax.plot(r["hist"], label=f"{name}, seed={r['seed']}", color=cfg["color"], linewidth=1.6)
ax.axhline(EPS, color="gray", linestyle="--", linewidth=0.8, label="порог Ee")
ax.set_yscale("log")
ax.set_xlabel("Эпоха")
ax.set_ylabel("Суммарная ошибка Es (BCE)")
ax.set_title("Сходимость: Сигмоида (А) vs ReLU (Б) на скрытом слое")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.35
x_pos = np.arange(len(SEEDS))
for i, (name, cfg) in enumerate(CONFIGS.items()):
    lst = runs[name]
    eps_vals = [r["epochs"] for r in lst]
    colors = [cfg["color"] if r["converged"] else "#cccccc" for r in lst]
    hatches = [None if r["converged"] else "//" for r in lst]
    bars = ax.bar(x_pos + (i - 0.5) * width, eps_vals, width, label=name, color=colors, edgecolor="black")
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"seed={s}" for s in SEEDS])
ax.set_ylabel("Эпох до критерия остановки")
ax.set_title("Устойчивость сходимости по 5 запускам (серые/штрих = не сошёлся)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("stability.png", dpi=150)
plt.close(fig)

grid = np.linspace(-10, 10, 200)
GA, GB = np.meshgrid(grid, grid)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, cfg) in zip(axes, CONFIGS.items()):
    model = rep[name]["model"]
    Z = np.zeros_like(GA)
    for i in range(GA.shape[0]):
        for j in range(GA.shape[1]):
            x = scaler.norm(np.array([GA[i, j], GB[i, j]]))
            Z[i, j] = model.predict(x)
    cf = ax.contourf(GA, GB, Z, levels=20, cmap="coolwarm")
    fig.colorbar(cf, ax=ax, label="ŷ (норм., 0..1)")
    colors_pts = ["#1d4ed8" if t == c0 else "#ea580c" for t in E_raw]
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=colors_pts, s=120, edgecolor="black", zorder=5)
    for (a, b), t in zip(X_raw, E_raw):
        ax.annotate(f"{t:.0f}", (a, b), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title(f"Разделяющая поверхность, {name} (seed={rep[name]['seed']})")
fig.tight_layout()
fig.savefig("decision_surface.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for row, name in enumerate(CONFIGS.keys()):
    model = rep[name]["model"]
    for col in range(2):
        ax = axes[row, col]
        Z = np.zeros_like(GA)
        for i in range(GA.shape[0]):
            for j in range(GA.shape[1]):
                xn = scaler.norm(np.array([GA[i, j], GB[i, j]]))
                s_hidden = xn @ model.W1 + model.T1
                yh = relu(s_hidden) if model.hidden == "relu" else sigmoid(s_hidden)
                Z[i, j] = yh[col]
        cf = ax.contourf(GA, GB, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax)
        ax.set_title(f"{name}, нейрон скрытого слоя {col + 1}")
        ax.set_xlabel("A")
        ax.set_ylabel("B")
fig.suptitle("Активация отдельных нейронов скрытого слоя: гладкая (сигмоида) vs кусочно-линейная (ReLU)")
fig.tight_layout()
fig.savefig("hidden_activations.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
names = list(CONFIGS.keys())
maes = [rep[n]["mae"] for n in names]
colors = [CONFIGS[n]["color"] for n in names]
ax.bar(names, maes, color=colors, edgecolor="black")
ax.set_ylabel("Средняя абсолютная ошибка в шкале [c0; c1]")
ax.set_title("Точность восстановления исходной шкалы\n(по сошедшимся представительным запускам)")
for i, v in enumerate(maes):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("mae_comparison.png", dpi=150)
plt.close(fig)

print("\nГрафики сохранены: convergence.png, stability.png, decision_surface.png, hidden_activations.png, mae_comparison.png")

demo_model = rep["A (Sigmoid)"]["model"]
print("\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ (конфигурация А, Сигмоида)")
demo_pairs = list(zip(X_raw.tolist(), E_raw.tolist())) + [([0, 0], None), ([10, -10], None), ([-3, 4], None)]
for (a, b), e_true in demo_pairs:
    x = scaler.norm(np.array([a, b], dtype=float))
    y_norm = float(demo_model.predict(x))
    y_real = float(scaler.denorm(y_norm))
    cls = scaler.nearest_class(y_real)
    tag = f"(ожид. {e_true:.0f})" if e_true is not None else "(вне выборки)"
    print(f"  A={a:>6.1f} B={b:>6.1f} -> y_hat_norm={y_norm:.4f}  y_hat_real={y_real:>8.3f}  класс={cls:>5.1f} {tag}")

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
    y_norm = float(demo_model.predict(x))
    y_real = float(scaler.denorm(y_norm))
    cls = scaler.nearest_class(y_real)
    print(f"  y_hat_norm={y_norm:.4f}  y_hat_real={y_real:.4f}  ближе к c={cls:.0f}")