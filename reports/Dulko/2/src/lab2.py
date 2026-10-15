from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C0, C1 = 7, -7
DATA = [(C0, C0, C0), (C0, C1, C1), (C1, C0, C1), (C1, C1, C0)]
X = np.array([[a / 10, b / 10] for a, b, _ in DATA])
Y = np.array([y for _, _, y in DATA], float)
TARGETS = np.array([0.0 if y == C0 else 1.0 for y in Y])
LR_MSE, LR_BCE = 0.5, 0.3
EE_MSE, EE_BCE = 0.005, 0.01
MAX_EPOCHS = 10000
SEEDS = [1, 2, 3, 4, 5]
COLORS = {"MSE": "tab:blue", "BCE": "tab:red"}
OUT = Path(__file__).resolve().parent / "figures"


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -500, 500)))


def to_real(output):
    return C0 + output * (C1 - C0)


def classify(value):
    return C0 if abs(value - C0) < abs(value - C1) else C1


class MLP:
    def __init__(self, rng):
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2)
        self.b2 = rng.uniform(-0.5, 0.5)

    def predict(self, x):
        self.hidden = sigmoid(x @ self.W1 + self.b1)
        return sigmoid(self.hidden @ self.W2 + self.b2)

    def train_step(self, x, target, learning_rate, loss):
        output = self.predict(x)
        delta_output = output - target
        if loss == "MSE":
            delta_output = (output - target) * output * (1 - output)
        delta_hidden = delta_output * self.W2 * self.hidden * (1 - self.hidden)
        self.W2 -= learning_rate * delta_output * self.hidden
        self.b2 -= learning_rate * delta_output
        self.W1 -= learning_rate * np.outer(x, delta_hidden)
        self.b1 -= learning_rate * delta_hidden


def total_error(network, loss):
    if loss == "MSE":
        return 0.5 * sum((t - network.predict(x)) ** 2 for x, t in zip(X, TARGETS))
    error = 0.0
    for x, target in zip(X, TARGETS):
        output = np.clip(network.predict(x), 1e-12, 1 - 1e-12)
        error -= target * np.log(output) + (1 - target) * np.log(1 - output)
    return float(error)


def train(loss, seed, learning_rate, threshold):
    rng = np.random.default_rng(seed)
    network = MLP(rng)
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(len(X)):
            network.train_step(X[i], TARGETS[i], learning_rate, loss)
        error = total_error(network, loss)
        history.append(error)
        if error <= threshold:
            break
    outputs = np.array([to_real(network.predict(x)) for x in X])
    accuracy = np.mean([classify(output) == target for output, target in zip(outputs, Y)])
    return dict(seed=seed, network=network, epochs=epoch, error=error, converged=error <= threshold,
                history=history, mae=float(np.mean(np.abs(outputs - Y))), accuracy=float(accuracy))


def representative(runs):
    successful = [run for run in runs if run["converged"]]
    candidates = sorted(successful or runs, key=lambda run: run["epochs"])
    return candidates[len(candidates) // 2]


def save_figure(figure, filename):
    figure.tight_layout()
    figure.savefig(OUT / filename, dpi=150)
    plt.close(figure)


def plot_results(results, selected):
    figure, axis = plt.subplots(figsize=(8, 5))
    for loss, threshold in [("MSE", EE_MSE), ("BCE", EE_BCE)]:
        run = selected[loss]
        axis.plot(range(1, len(run["history"]) + 1), run["history"],
                  color=COLORS[loss], label=f"{loss}, seed={run['seed']}")
        axis.axhline(threshold, color=COLORS[loss], ls=":", label=f"Ee {loss} = {threshold}")
    axis.set_yscale("log")
    axis.set(xlabel="Эпоха", ylabel="Суммарная ошибка Es", title="Сходимость MSE и BCE")
    axis.legend()
    axis.grid(alpha=0.3, which="both")
    save_figure(figure, "fig1_convergence.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(SEEDS))
    width = 0.35
    for offset, (loss, runs) in enumerate(results.items()):
        for x, run in zip(positions + (offset - 0.5) * width, runs):
            axis.bar(x, run["epochs"], width, color=COLORS[loss], edgecolor="black",
                     hatch=None if run["converged"] else "//")
            axis.text(x, run["epochs"], str(run["epochs"]), ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions, [str(seed) for seed in SEEDS])
    axis.set(xlabel="Seed", ylabel="Число эпох", title="Число эпох по пяти запускам", ylim=(0, None))
    legend = [Patch(color=COLORS[loss], label=loss) for loss in results]
    if any(not run["converged"] for runs in results.values() for run in runs):
        legend.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="не сошлась"))
    axis.legend(handles=legend)
    axis.grid(alpha=0.3, axis="y")
    save_figure(figure, "fig2_epochs.png")

    figure = plt.figure(figsize=(10, 8))
    layout = figure.add_gridspec(2, 2, height_ratios=[2, 1])
    axes = [figure.add_subplot(layout[0, 0]), figure.add_subplot(layout[0, 1])]
    values = np.linspace(-10, 10, 200)
    a_grid, b_grid = np.meshgrid(values, values)
    inputs = np.stack([a_grid, b_grid], axis=-1) / 10
    low, high = min(C0, C1), max(C0, C1)
    for axis, (loss, run) in zip(axes, selected.items()):
        outputs = to_real(run["network"].predict(inputs))
        image = axis.contourf(a_grid, b_grid, outputs, levels=np.linspace(low, high, 23), cmap="RdYlBu")
        axis.contour(a_grid, b_grid, outputs, levels=[(C0 + C1) / 2], colors="black", linewidths=1.5)
        for a, b, target in DATA:
            axis.scatter(a, b, s=140, c="tab:green" if target == C0 else "tab:purple",
                         marker="o" if target == C0 else "s", edgecolors="black", zorder=3)
        axis.set(xlabel="A", ylabel="B", title=f"{loss}, seed={run['seed']}")
        figure.colorbar(image, ax=axis, label="Выход сети")
    axes[0].legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color="tab:green", label=f"класс c0 ({C0})"),
        plt.Line2D([], [], marker="s", ls="", color="tab:purple", label=f"класс c1 ({C1})"),
        plt.Line2D([], [], color="black", label="граница классов")], fontsize=8)
    axis = figure.add_subplot(layout[1, :])
    averages = [np.mean([run["mae"] for run in runs]) for runs in results.values()]
    bars = axis.bar(list(results), averages, color=list(COLORS.values()))
    for bar, value in zip(bars, averages):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    axis.set(ylabel="MAE в исходной шкале", title="Средняя MAE пяти запусков", ylim=(0, max(averages) * 1.2))
    save_figure(figure, "fig3_surface_mae.png")


def show_prediction(network, a, b):
    output = float(network.predict(np.array([a, b]) / 10))
    real_output = to_real(output)
    label = "c0" if classify(real_output) == C0 else "c1"
    print(f"A={a:>6}, B={b:>6} | ŷ(0;1)={output:.4f} | в шкале [c0;c1]: {real_output:8.4f} | ближе к {label}")


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    print(f"Ee: MSE = {EE_MSE}, BCE = {EE_BCE}; lr MSE = {LR_MSE}, lr BCE = {LR_BCE}; max эпох = {MAX_EPOCHS}\n")
    for loss, learning_rate, threshold in [("MSE", LR_MSE, EE_MSE), ("BCE", LR_BCE, EE_BCE)]:
        runs = [train(loss, seed, learning_rate, threshold) for seed in SEEDS]
        results[loss] = runs
        print(f"Конфигурация {loss}")
        print(f"{'seed':>4} {'эпох':>7} {'Es':>10} {'сошлась':>8} {'accuracy':>9} {'MAE':>8}")
        for run in runs:
            status = "да" if run["converged"] else "НЕТ"
            print(f"{run['seed']:>4} {run['epochs']:>7} {run['error']:>10.5f} {status:>8} {run['accuracy']*100:>8.0f}% {run['mae']:>8.3f}")
        print("Сошлось:", sum(run["converged"] for run in runs), "из", len(runs), "\n")
    selected = {loss: representative(runs) for loss, runs in results.items()}
    print("Представительные запуски:")
    for loss, run in selected.items():
        print(f"{loss}: seed={run['seed']}, эпох={run['epochs']}")
    plot_results(results, selected)
    network = selected["BCE"]["network"]
    print("\nРежим функционирования (BCE)\nОбучающие примеры:")
    for a, b, _ in DATA:
        show_prediction(network, a, b)
    print("Дополнительные пары:")
    for a, b in [(6, -6), (-6, 6), (0, 0)]:
        show_prediction(network, a, b)
    print("\nДалее ввод своих значений. Пустая строка: выход.")
    while True:
        try:
            text = input("Введите A и B из [-10; 10] через пробел: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            break
        try:
            a, b = map(float, text.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                raise ValueError
            show_prediction(network, a, b)
        except ValueError:
            print("Некорректный ввод.")
    return results, selected


if __name__ == "__main__":
    main()
