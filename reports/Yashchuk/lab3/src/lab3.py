import numpy as np
import matplotlib.pyplot as plt

C0 = -5.0
C1 = 10.0

EE = 0.01
MAX_EPOCHS = 50000
LR_SIGMOID = 2.0
LR_RELU = 0.1
LEAKY_ALPHA = 0.01
SEEDS = [1, 2, 3, 4, 5]

X = np.array([
    [-5.0, -5.0],
    [-5.0, 10.0],
    [10.0, -5.0],
    [10.0, 10.0]
])
Y = np.array([[0.0], [1.0], [1.0], [0.0]])


def normalize_input(x):
    return np.asarray(x, dtype=float) / 10.0


def denormalize_output(y):
    return C0 + np.asarray(y) * (C1 - C0)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def leaky_relu(x):
    return np.where(x > 0, x, LEAKY_ALPHA * x)


def leaky_relu_deriv(x):
    return np.where(x > 0, 1.0, LEAKY_ALPHA)


def initialize_network(seed, activation):
    rng = np.random.default_rng(seed)

    w = rng.normal(0, 0.15, (1, 2))
    W1 = np.vstack([w, w])

    if activation == "relu":
        b1 = rng.uniform(0.5, 1.5, (1, 2))
    else:
        b1 = rng.normal(0, 0.05, (1, 2))

    W2 = rng.normal(0, 0.15, (2, 1))
    b2 = rng.normal(0, 0.05, (1, 1))

    return W1, b1, W2, b2


def forward(x, weights, activation):
    W1, b1, W2, b2 = weights
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, 2)

    z1 = x @ W1 + b1
    h = leaky_relu(z1) if activation == "relu" else sigmoid(z1)
    z2 = h @ W2 + b2
    return sigmoid(z2), z1, h


def forward_symmetric(x, weights, activation):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, 2)
    out1, _, _ = forward(x, weights, activation)
    out2, _, _ = forward(x[:, ::-1], weights, activation)
    return (out1 + out2) / 2.0


def calculate_error(x, y, weights, activation):
    output = forward_symmetric(x, weights, activation)
    output = np.clip(output, 1e-12, 1.0 - 1e-12)
    errors = -(y * np.log(output) + (1.0 - y) * np.log(1.0 - output))
    return float(np.sum(errors))


def train_network(seed, activation):
    weights = list(initialize_network(seed, activation))
    lr = LR_RELU if activation == "relu" else LR_SIGMOID
    rng = np.random.default_rng(seed + 100)

    x_train = normalize_input(X)
    history = []
    converged = False

    for epoch in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(len(x_train)):
            x = x_train[i:i + 1]
            target = Y[i:i + 1]
            W1, b1, W2, b2 = weights

            grads = []
            for xin in (x, x[:, ::-1]):
                output, z1, h = forward(xin, weights, activation)
                delta_out = output - target

                grad_W2 = h.T @ delta_out
                grad_b2 = np.sum(delta_out, axis=0, keepdims=True)

                delta_hidden = delta_out @ W2.T
                if activation == "relu":
                    delta_hidden *= leaky_relu_deriv(z1)
                else:
                    delta_hidden *= h * (1.0 - h)

                grad_W1 = xin.T @ delta_hidden
                grad_b1 = np.sum(delta_hidden, axis=0, keepdims=True)
                grads.append((grad_W1, grad_b1, grad_W2, grad_b2))

            gW1 = (grads[0][0] + grads[1][0]) / 2
            gb1 = (grads[0][1] + grads[1][1]) / 2
            gW2 = (grads[0][2] + grads[1][2]) / 2
            gb2 = (grads[0][3] + grads[1][3]) / 2

            weights[0] -= lr * gW1
            weights[1] -= lr * gb1
            weights[2] -= lr * gW2
            weights[3] -= lr * gb2

        error = calculate_error(x_train, Y, weights, activation)
        history.append(error)

        if error <= EE:
            converged = True
            break

    output = forward_symmetric(x_train, weights, activation)
    output_original = denormalize_output(output)
    predicted_classes = (output >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes == Y)

    expected_original = np.array([[C0], [C1], [C1], [C0]])
    mae = np.mean(np.abs(output_original - expected_original))

    return {
        "weights": weights,
        "activation": activation,
        "seed": seed,
        "epochs": epoch,
        "error": history[-1],
        "history": history,
        "accuracy": accuracy,
        "mae": float(mae),
        "converged": converged
    }


def print_results(name, results):
    print("\n" + name)
    print("-" * 76)
    print(f"{'Seed':<8}{'Эпохи':<12}{'Ошибка':<15}"
          f"{'Accuracy':<14}{'MAE':<12}{'Критерий':<12}")
    for r in results:
        status = "достигнут" if r["converged"] else "не достигнут"
        print(f"{r['seed']:<8}{r['epochs']:<12}"
              f"{r['error']:<15.6f}{r['accuracy']:<14.2f}"
              f"{r['mae']:<12.4f}{status:<12}")
    successful = [r["epochs"] for r in results if r["converged"]]
    print(f"Успешных запусков: {len(successful)} из {len(results)}")
    if successful:
        print(f"Минимум эпох: {min(successful)}")
        print(f"Максимум эпох: {max(successful)}")
        print(f"Среднее число эпох: {np.mean(successful):.1f}")
    print(f"Средняя MAE: {np.mean([r['mae'] for r in results]):.4f}")


def plot_convergence(results_sigmoid, results_relu):
    best_sig = results_sigmoid[0]
    best_relu = results_relu[0]

    plt.figure(figsize=(9, 5))
    plt.plot(best_sig["history"],
             label=f"Сигмоида (seed={best_sig['seed']}, {best_sig['epochs']} эпох)")
    plt.plot(best_relu["history"],
             label=f"ReLU (seed={best_relu['seed']}, {best_relu['epochs']} эпох)")
    plt.axhline(EE, color="black", linestyle="--", label=f"Порог Ee = {EE}")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка BCE")
    plt.title("Сравнение сходимости сетей")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_epochs(results_sigmoid, results_relu):
    seeds = [r["seed"] for r in results_sigmoid]
    epochs_sigmoid = [r["epochs"] for r in results_sigmoid]
    epochs_relu = [r["epochs"] for r in results_relu]
    positions = np.arange(len(seeds))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(positions - width / 2, epochs_sigmoid, width, label="Сигмоида")
    plt.bar(positions + width / 2, epochs_relu, width, label="ReLU")
    plt.xticks(positions, seeds)
    plt.xlabel("Seed")
    plt.ylabel("Число эпох")
    plt.title("Число эпох в пяти запусках")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()


def plot_surfaces(result_sigmoid, result_relu):
    grid = np.linspace(-10, 10, 101)
    A, B = np.meshgrid(grid, grid)
    points = np.column_stack((A.ravel(), B.ravel()))
    points_normalized = normalize_input(points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, result, title in [
        (axes[0], result_sigmoid, "Скрытый слой: сигмоида"),
        (axes[1], result_relu, "Скрытый слой: ReLU")
    ]:
        output = forward_symmetric(points_normalized,
                                   result["weights"], result["activation"])
        Z = denormalize_output(output).reshape(A.shape)
        contour = ax.contourf(A, B, Z, levels=30, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(),
                   cmap="coolwarm", edgecolors="black", s=90)
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_title(title)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        fig.colorbar(contour, ax=ax, label="Выход сети")
    fig.suptitle("Выходные поверхности сетей")
    fig.tight_layout()


def plot_mae(results_sigmoid, results_relu):
    mae_sigmoid = np.mean([r["mae"] for r in results_sigmoid])
    mae_relu = np.mean([r["mae"] for r in results_relu])
    plt.figure(figsize=(7, 5))
    plt.bar(["Сигмоида", "ReLU"], [mae_sigmoid, mae_relu])
    plt.ylabel("Средняя абсолютная ошибка MAE")
    plt.title("Сравнение MAE по пяти запускам")
    plt.grid(axis="y")
    plt.tight_layout()


def demonstrate(result, name):
    print(f"\nПроверка сети: {name}")
    print("A       B       y_norm   y         Класс    Ожидаемый")
    expected = np.array([C0, C1, C1, C0])
    for i, (a, b) in enumerate(X):
        output = forward_symmetric(normalize_input([a, b]),
                                   result["weights"], result["activation"])
        y_norm = float(output[0, 0])
        y_value = float(denormalize_output(y_norm))
        predicted_class = C0 if abs(y_value - C0) <= abs(y_value - C1) else C1
        print(f"{a:<7.1f} {b:<7.1f} {y_norm:<8.4f} "
              f"{y_value:<9.4f} {predicted_class:<8.1f} {expected[i]:.1f}")
    print("\nДополнительные пары:")
    for a, b in [(0, 0), (5, 5), (-6, 6)]:
        output = forward_symmetric(normalize_input([a, b]),
                                   result["weights"], result["activation"])
        y_norm = float(output[0, 0])
        y_value = float(denormalize_output(y_norm))
        predicted_class = C0 if abs(y_value - C0) <= abs(y_value - C1) else C1
        print(f"A={a:>5.1f}, B={b:>5.1f} | "
              f"y_norm={y_norm:.4f} | y={y_value:.4f} | класс={predicted_class:.1f}")


def interactive_mode(result):
    print("\nИнтерактивный режим")
    print("Введите два числа A и B от -10 до 10.")
    print("Для завершения введите q.")
    while True:
        text = input("A B: ").strip()
        if text.lower() == "q":
            break
        try:
            values = text.split()
            if len(values) != 2:
                raise ValueError
            a, b = map(float, values)
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                print("Оба числа должны быть в диапазоне [-10; 10].")
                continue
            output = forward_symmetric(normalize_input([a, b]),
                                       result["weights"], result["activation"])
            y_norm = float(output[0, 0])
            y_value = float(denormalize_output(y_norm))
            predicted_class = C0 if abs(y_value - C0) <= abs(y_value - C1) else C1
            print(f"Нормализованный выход: {y_norm:.4f}")
            print(f"Выход в исходной шкале: {y_value:.4f}")
            print(f"Ближайший класс: {predicted_class:.1f}")
        except ValueError:
            print("Ошибка ввода. Пример: -3 7")


def main():
    print("Лабораторная работа №3")
    print("Вариант 14: c0 =", C0, ", c1 =", C1)
    print("Архитектура MLP 2-2-1, функция потерь BCE")

    results_sigmoid = [train_network(seed, "sigmoid") for seed in SEEDS]
    results_relu = [train_network(seed, "relu") for seed in SEEDS]

    print_results("Конфигурация A: сигмоида на скрытом слое", results_sigmoid)
    print_results("Конфигурация B: ReLU на скрытом слое", results_relu)

    print("\nСравнение средних результатов")
    print(f"{'Показатель':<25}{'Сигмоида':<15}{'ReLU':<15}")
    print(f"{'Средняя MAE':<25}"
          f"{np.mean([r['mae'] for r in results_sigmoid]):<15.4f}"
          f"{np.mean([r['mae'] for r in results_relu]):<15.4f}")
    print(f"{'Средняя ошибка':<25}"
          f"{np.mean([r['error'] for r in results_sigmoid]):<15.6f}"
          f"{np.mean([r['error'] for r in results_relu]):<15.6f}")

    results_sigmoid.sort(key=lambda r: (not r["converged"], r["epochs"]))
    results_relu.sort(key=lambda r: (not r["converged"], r["epochs"]))

    plot_convergence(results_sigmoid, results_relu)
    plot_epochs(results_sigmoid, results_relu)
    plot_surfaces(results_sigmoid[0], results_relu[0])
    plot_mae(results_sigmoid, results_relu)

    demonstrate(results_sigmoid[0], "Сигмоида")
    demonstrate(results_relu[0], "ReLU")

    plt.show()

    interactive_mode(results_relu[0])


if __name__ == "__main__":
    main()