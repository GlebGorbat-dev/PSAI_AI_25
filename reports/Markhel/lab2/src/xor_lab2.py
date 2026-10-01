
import math
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

C0, C1 = -9.0, 6.0
LO, HI = -10.0, 10.0

X_RAW = np.array([[-9.0, -9.0],
                   [-9.0,  6.0],
                   [ 6.0, -9.0],
                   [ 6.0,  6.0]])
T_RAW = np.array([-9.0, 6.0, 6.0, -9.0])


def normalize(x):
    """Нормировка по границам диапазона [-10; 10] -> [0; 1] (входы сети и цели конфигурации А, как в ЛР №1)."""
    return (np.asarray(x, dtype=float) - LO) / (HI - LO)


def denormalize(y):
    return np.asarray(y, dtype=float) * (HI - LO) + LO


# НОВОЕ (ЛР №2): нормализация целей для BCE: c0 -> 0, c1 -> 1, и обратный пересчёт выхода в шкалу [c0; c1]
def to_binary(y_real):
    return (np.asarray(y_real, dtype=float) - C0) / (C1 - C0)


def from_binary(y_hat):
    return C0 + np.asarray(y_hat, dtype=float) * (C1 - C0)


def to_real(y_hat, loss):
    """Выход сети ŷ из (0; 1) -> исходная шкала [c0; c1] (способ зависит от конфигурации)."""
    return denormalize(y_hat) if loss == "mse" else from_binary(y_hat)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(y):
    """Производная сигмоиды y=sigmoid(x)."""
    return y * (1.0 - y)


X = normalize(X_RAW)
T_MSE = normalize(T_RAW)      # конфигурация А: 0.05 / 0.80
T_BCE = to_binary(T_RAW)      # конфигурация Б: 0 / 1

LR = 0.5
MAX_EPOCHS = 50000
EE_MSE = 0.001                # порог для MSE - то же значение, что в ЛР №1


def equivalent_ee_bce(ee_mse=EE_MSE, n=4):
    """
    Порог Ee для BCE, эквивалентный по точности порогу Ee_MSE.
    Ee_MSE означает ошибку e = (HI - LO) * sqrt(Ee_MSE / n) в шкале [c0; c1] на каждом примере.
    В шкале ŷ это eps = e / (c1 - c0); вклад одного примера в BCE равен -ln(1 - eps), всего n примеров.
    """
    eps = (HI - LO) * math.sqrt(ee_mse / n) / (C1 - C0)
    return n * (-math.log(1.0 - eps))


EE_BCE = equivalent_ee_bce()

CONFIGS = {
    "A": {"name": "А: MSE", "loss": "mse", "T": T_MSE, "Ee": EE_MSE},
    "B": {"name": "Б: BCE", "loss": "bce", "T": T_BCE, "Ee": EE_BCE},
}


def bce_value(o, t):
    o = np.clip(o, 1e-12, 1.0 - 1e-12)          # защита от log(0)
    return float(-np.sum(t * np.log(o) + (1.0 - t) * np.log(1.0 - o)))


def train_loop(model, X, T, Ee, max_epochs, rng):
    """Общий цикл обучения (онлайн): Es накапливается по примерам внутри эпохи, как в ЛР №1."""
    rng = rng or np.random.default_rng(0)
    n = len(X)
    Es = float("inf")
    history = []
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n)
        Es = 0.0
        for i in order:
            Es += model.train_step(X[i], T[i:i + 1])
        history.append(Es)
        if Es <= Ee:
            return epoch, Es, history
    return max_epochs, Es, history


class MLP221:
    def __init__(self, lr=0.5, seed=None, loss="mse"):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, (2,))
        self.W2 = rng.uniform(-0.5, 0.5, (2, 1))
        self.b2 = rng.uniform(-0.5, 0.5, (1,))
        self.lr = lr
        self.loss = loss          # НОВОЕ: "mse" (ЛР №1) или "bce"

    def forward(self, x):
        self.h_in = x @ self.W1 + self.b1
        self.h_out = sigmoid(self.h_in)
        self.o_in = self.h_out @ self.W2 + self.b2
        self.o_out = sigmoid(self.o_in)
        return self.o_out

    def train_step(self, x, t):
        o = self.forward(x)
        err = t - o
        if self.loss == "mse":
            delta_out = err * dsigmoid_from_output(o)       # MSE + сигмоида (ЛР №1)
            value = float(np.sum(err ** 2))
        else:
            delta_out = err                                  # НОВОЕ: BCE + сигмоида (o(1-o) сокращается)
            value = bce_value(o, t)
        delta_hidden = (delta_out @ self.W2.T) * dsigmoid_from_output(self.h_out)   # скрытый слой: без изменений

        self.W2 += self.lr * np.outer(self.h_out, delta_out)
        self.b2 += self.lr * delta_out
        self.W1 += self.lr * np.outer(x, delta_hidden)
        self.b1 += self.lr * delta_hidden
        return value

    def train(self, X, T, Ee=0.001, max_epochs=50000, rng=None):
        return train_loop(self, X, T, Ee, max_epochs, rng)

    def predict(self, x):
        return float(self.forward(np.asarray(x, dtype=float)).item())


class SingleLayerPerceptron:
    """Однослойный персептрон из ЛР №1 (только для проверки линейной неразделимости при BCE и MSE)."""

    def __init__(self, lr=0.5, seed=None, loss="mse"):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.5, 0.5, (2, 1))
        self.b = rng.uniform(-0.5, 0.5, (1,))
        self.lr = lr
        self.loss = loss

    def forward(self, x):
        self.o_in = x @ self.W + self.b
        self.o_out = sigmoid(self.o_in)
        return self.o_out

    def train_step(self, x, t):
        o = self.forward(x)
        err = t - o
        if self.loss == "mse":
            delta = err * dsigmoid_from_output(o)
            value = float(np.sum(err ** 2))
        else:
            delta = err
            value = bce_value(o, t)
        self.W += self.lr * np.outer(x, delta)
        self.b += self.lr * delta
        return value

    def train(self, X, T, Ee=0.001, max_epochs=50000, rng=None):
        return train_loop(self, X, T, Ee, max_epochs, rng)

    def predict(self, x):
        return float(self.forward(np.asarray(x, dtype=float)).item())


def classify(y_real):

    return C0 if abs(y_real - C0) <= abs(y_real - C1) else C1


def evaluate(model, X, T_RAW):
    """Выходы в шкале [c0; c1], accuracy и средняя абсолютная ошибка (MAE) в шкале [c0; c1]."""
    outputs_hat, outputs_real = [], []
    correct = 0
    for x, t_raw in zip(X, T_RAW):
        y_hat = model.predict(x)
        y_real = float(to_real(y_hat, model.loss))
        outputs_hat.append(y_hat)
        outputs_real.append(y_real)
        if classify(y_real) == t_raw:
            correct += 1
    accuracy = correct / len(X)
    mae = float(np.mean(np.abs(np.array(outputs_real) - T_RAW)))
    return outputs_hat, outputs_real, accuracy, mae


def run_experiment(name, model, T, Ee, max_epochs, seed, verbose=True):
    rng = np.random.default_rng(seed)
    epochs, Es, history = model.train(X, T, Ee=Ee, max_epochs=max_epochs, rng=rng)
    outputs_hat, outputs_real, accuracy, mae = evaluate(model, X, T_RAW)
    converged = Es <= Ee

    if verbose:
        print(f"=== {name} ===")
        print(f"Критерий остановки Ee = {Ee:.4f}, максимум эпох = {max_epochs}")
        print(f"Эпох затрачено: {epochs}  |  Сходимость достигнута: {'да' if converged else 'нет'}")
        print(f"Итоговая суммарная ошибка Es = {Es:.6f}")
        print(f"{'A':>6}{'B':>6}{'A xor B (цель)':>16}{'ŷ':>10}{'выход сети':>14}{'класс':>8}")
        for (a, b), t_raw, y_hat, y_real in zip(X_RAW, T_RAW, outputs_hat, outputs_real):
            print(f"{a:6.0f}{b:6.0f}{t_raw:16.0f}{y_hat:10.4f}{y_real:14.3f}{classify(y_real):8.0f}")
        print(f"Accuracy = {accuracy * 100:.1f}%")
        print(f"Средняя абсолютная ошибка в шкале [c0; c1] = {mae:.4f}")
        print()
    return {"name": name, "model": model, "seed": seed, "Ee": Ee, "epochs": epochs, "Es": Es,
            "converged": converged, "accuracy": accuracy, "mae": mae, "history": np.array(history),
            "outputs_hat": outputs_hat, "outputs_real": outputs_real}


def run_config(key, seed, verbose=False):
    cfg = CONFIGS[key]
    model = MLP221(lr=LR, seed=seed, loss=cfg["loss"])
    return run_experiment(cfg["name"], model, cfg["T"], cfg["Ee"], MAX_EPOCHS, seed, verbose)


def run_series(key, seeds):
    """Серия запусков с разной случайной инициализацией весов (и порядка примеров)."""
    return [run_config(key, s) for s in seeds]


def pick_representative(runs):
    """Представительный запуск: среди сошедшихся - с медианным числом эпох."""
    ok = sorted((r for r in runs if r["converged"]), key=lambda r: r["epochs"])
    return ok[len(ok) // 2] if ok else runs[0]


def print_series(runs_a, runs_b):
    for key, runs in (("A", runs_a), ("B", runs_b)):
        print(f"=== Серия запусков: конфигурация {runs[0]['name']} (Ee = {runs[0]['Ee']:.4f}) ===")
        print(f"{'seed':>5}{'эпох':>8}{'сошлась':>9}{'Es':>12}{'accuracy':>10}{'MAE':>9}")
        for r in runs:
            print(f"{r['seed']:>5}{r['epochs']:>8}{'да' if r['converged'] else 'нет':>9}"
                  f"{r['Es']:>12.6f}{r['accuracy'] * 100:>9.0f}%{r['mae']:>9.4f}")
        ep = [r["epochs"] for r in runs if r["converged"]]
        if ep:
            print(f"Сошлось {len(ep)} из {len(runs)}; эпох: min={min(ep)}, max={max(ep)}, "
                  f"среднее={np.mean(ep):.0f}, медиана={np.median(ep):.0f}, std={np.std(ep):.0f}")
        else:
            print(f"Сошлось 0 из {len(runs)}")
        print()


def extra_study(seeds=range(100, 150)):
    """Дополнительная серия из 50 запусков: устойчивость к инициализации + BCE с тем же Ee=0.001, что у MSE."""
    cfgs = [("А: MSE", "mse", T_MSE, EE_MSE), ("Б: BCE", "bce", T_BCE, EE_BCE),
            ("Б: BCE, Ee=0.001", "bce", T_BCE, EE_MSE)]
    out = {}
    for name, loss, T, Ee in cfgs:
        runs = [run_experiment(name, MLP221(lr=LR, seed=s, loss=loss), T, Ee, MAX_EPOCHS, s, verbose=False)
                for s in seeds]
        ok = np.array([r["epochs"] for r in runs if r["converged"]])
        out[name] = {
            "n": len(runs), "converged": int(len(ok)), "Ee": Ee,
            "min": int(ok.min()), "max": int(ok.max()), "mean": float(ok.mean()),
            "median": float(np.median(ok)), "std": float(ok.std()),
            "mae": float(np.mean([r["mae"] for r in runs if r["converged"]])),
            "failed": [(r["seed"], r["Es"], r["accuracy"]) for r in runs if not r["converged"]],
        }
    return out


def print_extra(stats):
    print("=== Дополнительная серия: 50 запусков (seed 100-149) на конфигурацию ===")
    print(f"{'конфигурация':<20}{'Ee':>8}{'сошлось':>9}{'мин':>7}{'медиана':>9}{'среднее':>9}{'макс':>7}{'std':>6}{'MAE':>9}")
    for name, s in stats.items():
        print(f"{name:<20}{s['Ee']:>8.4f}{s['converged']:>6}/{s['n']:<3}{s['min']:>7}{s['median']:>9.0f}"
              f"{s['mean']:>9.0f}{s['max']:>7}{s['std']:>6.0f}{s['mae']:>9.4f}")
        for seed, es, acc in s["failed"]:
            print(f"    критерий не достигнут: seed={seed}, Es={es:.4f}, accuracy={acc * 100:.0f}%")
    print()


# ----------------------------------------------------------------------------------------
# Режим функционирования
# ----------------------------------------------------------------------------------------
def functioning(run, a, b):
    """Прямой проход обученной сети для пары (A, B): ŷ, значение в шкале [c0; c1], ближайший класс."""
    x = normalize(np.array([a, b]))
    y_hat = run["model"].predict(x)
    y_real = float(to_real(y_hat, run["model"].loss))
    return y_hat, y_real, classify(y_real)


def describe(run_a, run_b, a, b):
    """Строки, которые печатает интерактивный режим для пары (A, B)."""
    yh, yr, cl = functioning(run_b, a, b)
    yh2, yr2, cl2 = functioning(run_a, a, b)
    return [f"  Б (BCE): ŷ = {yh:.4f}  y = {yr:.3f}  -> ближе к классу {cl:.0f} ({'c0' if cl == C0 else 'c1'})",
            f"  А (MSE): ŷ = {yh2:.4f}  y = {yr2:.3f}  -> ближе к классу {cl2:.0f} ({'c0' if cl2 == C0 else 'c1'})"]


def demo_predictions(run_a, run_b, examples):
    print("=== Режим функционирования (проверочные примеры) ===")
    print(f"{'A':>6}{'B':>6}{'Б: ŷ':>9}{'Б: y':>9}{'Б: класс':>10}{'А: ŷ':>9}{'А: y':>9}{'А: класс':>10}")
    for a, b in examples:
        yh_b, yr_b, cl_b = functioning(run_b, a, b)
        yh_a, yr_a, cl_a = functioning(run_a, a, b)
        print(f"{a:6.1f}{b:6.1f}{yh_b:9.4f}{yr_b:9.3f}{cl_b:10.1f}{yh_a:9.4f}{yr_a:9.3f}{cl_a:10.1f}")
    print()


def interactive_loop(run_a, run_b):
    print("=== Интерактивный режим ===")
    print(f"Введите пару чисел A, B из диапазона [{LO:.0f}; {HI:.0f}] через пробел")
    print("(например: 3 -5). Для выхода введите 'q'.")
    while True:
        try:
            raw = input("A B > ").strip()
        except EOFError:
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Нужно ввести ровно два числа.")
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print("Не удалось распознать числа.")
            continue
        if not (LO <= a <= HI and LO <= b <= HI):
            print(f"Значения должны быть в диапазоне [{LO:.0f}; {HI:.0f}].")
            continue
        for line in describe(run_a, run_b, a, b):
            print(line)


# ----------------------------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------------------------
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLOR_A = "#1f77b4"    # синий - конфигурация А (MSE)
COLOR_B = "#e8710a"    # оранжевый - конфигурация Б (BCE)


def fig_convergence(rep_a, rep_b, path):
    """График сходимости: Es(эпоха) для А и Б на одних осях + пороги Ee; справа те же кривые в долях порога."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, norm in ((ax1, False), (ax2, True)):
        for rep, color in ((rep_a, COLOR_A), (rep_b, COLOR_B)):
            ep = np.arange(1, len(rep["history"]) + 1)
            y = rep["history"] / rep["Ee"] if norm else rep["history"]
            ax.plot(ep, y, color=color, lw=2, label=f"{rep['name']} (seed={rep['seed']}, {rep['epochs']} эпох)")
        if norm:
            ax.axhline(1.0, color="black", ls="--", lw=1.3, label="порог остановки Es = Ee")
        else:
            ax.axhline(EE_MSE, color=COLOR_A, ls="--", lw=1.3, label=f"Ee (MSE) = {EE_MSE:g}")
            ax.axhline(EE_BCE, color=COLOR_B, ls="--", lw=1.3, label=f"Ee (BCE) = {EE_BCE:.4f}")
        ax.set_yscale("log")
        ax.set_xlabel("Номер эпохи")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    ax1.set_ylabel("Суммарная ошибка Es (лог. шкала)")
    ax1.set_title("Сходимость: Es(эпоха), MSE и BCE")
    ax2.set_ylabel("Es / Ee (лог. шкала)")
    ax2.set_title("Те же кривые, нормированные на свой порог Ee")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def fig_epochs(runs_a, runs_b, path):
    """Число эпох по запускам (seed). Запуски, не достигшие критерия, - серая штриховка."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    seeds = [r["seed"] for r in runs_a]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for runs, dx, color in ((runs_a, -w / 2, COLOR_A), (runs_b, w / 2, COLOR_B)):
        for xi, r in zip(x, runs):
            ok = r["converged"]
            ax.bar(xi + dx, r["epochs"], w, color=color if ok else "lightgray",
                   edgecolor=color if ok else "black", hatch=None if ok else "//")
            ax.text(xi + dx, r["epochs"], f"{r['epochs']}" + ("" if ok else "\n(не сошлась)"),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_yscale("log")
    ax.set_ylim(top=MAX_EPOCHS * 3)
    ax.axhline(MAX_EPOCHS, color="gray", ls=":", lw=1)
    ax.text(len(seeds) - 0.5, MAX_EPOCHS * 1.05, f"лимит {MAX_EPOCHS} эпох", ha="right", va="bottom",
            fontsize=8, color="gray")
    ax.set_xlabel("Запуск (начальная инициализация весов)")
    ax.set_ylabel("Число эпох до Es ≤ Ee (лог. шкала)")
    ax.set_title(f"Разброс числа эпох по {len(seeds)} запускам")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(handles=[Patch(color=COLOR_A, label="А: MSE"), Patch(color=COLOR_B, label="Б: BCE"),
                       Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="критерий не достигнут")],
              loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def draw_surface(ax, fig, run, grid=241):
    """Карта выхода сети на плоскости (A, B) ∈ [-10; 10]^2 в общей шкале [c0; c1]."""
    from matplotlib.lines import Line2D
    model = run["model"]
    g = np.linspace(LO, HI, grid)
    AA, BB = np.meshgrid(g, g)
    pts = normalize(np.stack([AA.ravel(), BB.ravel()], axis=1))
    y_real = to_real(model.forward(pts).ravel(), model.loss).reshape(AA.shape)
    z_hidden = pts @ model.W1 + model.b1          # суммы скрытых нейронов

    im = ax.imshow(y_real, origin="lower", extent=[LO, HI, LO, HI], cmap="viridis", vmin=C0, vmax=C1, aspect="equal")
    mid = (C0 + C1) / 2.0
    ax.contour(AA, BB, y_real, levels=[mid], colors="black", linewidths=2.5)
    for j, ls in enumerate(("--", ":")):
        ax.contour(AA, BB, z_hidden[:, j].reshape(AA.shape), levels=[0.0], colors="red", linewidths=1.8,
                   linestyles=ls)
    for cls, face, marker in ((C0, "white", "o"), (C1, "black", "s")):
        m = T_RAW == cls
        ax.scatter(X_RAW[m, 0], X_RAW[m, 1], c=face, marker=marker, s=140, edgecolors="red", linewidths=1.8,
                   zorder=5)
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title(f"{run['name']}: выход сети в шкале [c0; c1]\n(seed={run['seed']}, {run['epochs']} эпох)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if model.loss == "bce":
        cb.set_label(f"y в шкале [c0; c1]   (ŷ = (y − {C0:g}) / {C1 - C0:g})")
    else:
        cb.set_label("y в шкале [c0; c1]   (y = 20·ŷ − 10)")
    return [Line2D([], [], color="black", lw=2.5, label=f"граница классов (y = {mid:g})"),
            Line2D([], [], color="red", lw=1.8, ls="--", label="скрытый нейрон h1: z = 0"),
            Line2D([], [], color="red", lw=1.8, ls=":", label="скрытый нейрон h2: z = 0"),
            Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor="red", markersize=9,
                   label=f"обучающие точки класса c0 = {C0:g}"),
            Line2D([], [], marker="s", color="w", markerfacecolor="black", markeredgecolor="red", markersize=9,
                   label=f"обучающие точки класса c1 = {C1:g}")]


def fig_surface_and_accuracy(runs_a, runs_b, rep_a, rep_b, path):
    """Subplot 2x2: разделяющие поверхности А и Б (сверху), точность восстановления шкалы (снизу)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.5, 11.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    handles = draw_surface(ax_a, fig, rep_a)
    draw_surface(ax_b, fig, rep_b)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)

    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    names, colors = ["А: MSE", "Б: BCE"], [COLOR_A, COLOR_B]
    vals = [rep_a["mae"], rep_b["mae"]]
    bars = ax1.bar(names, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
    ax1.set_ylabel("Средняя абсолютная ошибка в шкале [c0; c1]")
    ax1.set_title("MAE представительного запуска")
    ax1.grid(True, axis="y", alpha=0.3)

    means, stds = [], []
    for runs in (runs_a, runs_b):
        m = [r["mae"] for r in runs if r["converged"]]
        means.append(np.mean(m) if m else np.nan)
        stds.append(np.std(m) if m else 0.0)
    bars = ax2.bar(names, means, yerr=stds, capsize=6, color=colors, width=0.5)
    for b, v, e in zip(bars, means, stds):
        ax2.text(b.get_x() + b.get_width() / 2, v + e, f"{v:.4f}", ha="center", va="bottom")
    ax2.set_ylabel("Средняя абсолютная ошибка в шкале [c0; c1]")
    ax2.set_title(f"MAE: среднее ± std по сошедшимся из {len(runs_a)} запусков")
    ax2.grid(True, axis="y", alpha=0.3)
    ymax = max(max(vals), np.nanmax(means) + max(stds)) * 1.2
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)
    fig.suptitle("Разделяющая поверхность и точность восстановления шкалы [c0; c1]", fontsize=14)
    fig.savefig(path, dpi=150)
    return fig


def make_figures(runs_a, runs_b, rep_a, rep_b):
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = {"convergence": os.path.join(FIG_DIR, "fig1_convergence.png"),
             "epochs": os.path.join(FIG_DIR, "fig2_epochs_scatter.png"),
             "surface": os.path.join(FIG_DIR, "fig3_surface_accuracy.png")}
    fig_convergence(rep_a, rep_b, paths["convergence"])
    fig_epochs(runs_a, runs_b, paths["epochs"])
    fig_surface_and_accuracy(runs_a, runs_b, rep_a, rep_b, paths["surface"])
    return paths


SEEDS = [1, 2, 3, 4, 5]
DEMO_EXAMPLES = [(-9, -9), (-9, 6), (6, -9), (6, 6), (0, 0), (-10, 10), (3, -7), (-5, 8)]


if __name__ == "__main__":
    show = "--no-show" not in sys.argv

    if not show:
        import matplotlib
        matplotlib.use("Agg")

    print(f"Цели А (MSE): c0 -> {T_MSE[0]:.2f}, c1 -> {T_MSE[1]:.2f};  Б (BCE): c0 -> {T_BCE[0]:.0f}, c1 -> {T_BCE[1]:.0f}")
    print(f"Порог Ee: MSE = {EE_MSE}, BCE (эквивалентный по точности) = {EE_BCE:.4f}")
    print()

    # Представительные запуски в том же виде, что и в ЛР №1 (seed = 1)
    run_config("A", 1, verbose=True)
    run_config("B", 1, verbose=True)

    # Серия из 5 запусков с разной инициализацией весов
    runs_a = run_series("A", SEEDS)
    runs_b = run_series("B", SEEDS)
    print_series(runs_a, runs_b)
    rep_a, rep_b = pick_representative(runs_a), pick_representative(runs_b)
    print(f"Представительные запуски (медиана по эпохам): А - seed {rep_a['seed']}, Б - seed {rep_b['seed']}")
    print()

    # Однослойный персептрон: линейная неразделимость не зависит от функции потерь (вопросы 5 и 6)
    for loss, T, Ee in (("mse", T_MSE, EE_MSE), ("bce", T_BCE, EE_BCE)):
        slp = SingleLayerPerceptron(lr=LR, seed=1, loss=loss)
        run_experiment(f"Однослойный персептрон, {loss.upper()}", slp, T, Ee, MAX_EPOCHS, seed=1)

    if "--extra" in sys.argv:
        print_extra(extra_study())

    paths = make_figures(runs_a, runs_b, rep_a, rep_b)
    print("Графики сохранены:")
    for p in paths.values():
        print("  ", p)
    print()

    demo_predictions(rep_a, rep_b, DEMO_EXAMPLES)

    if sys.stdin.isatty():
        interactive_loop(rep_a, rep_b)
    else:
        print("(stdin не является интерактивным терминалом - интерактивный режим пропущен;"
              " запустите `python xor_lab2.py` в консоли, чтобы ввести свои A, B)")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
