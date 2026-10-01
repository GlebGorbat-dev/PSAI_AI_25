import numpy as np
import matplotlib.pyplot as plt

c0, c1 = -9, 6

data = np.array([
    [-9, -9, -9],
    [-9,  6,  6],
    [ 6, -9,  6],
    [ 6,  6, -9]
], dtype=float)

X = data[:, :2] / 10
Y = ((data[:, 2] - c0) / (c1 - c0)).reshape(-1, 1)

lr = 0.5
maxepoch = 5000
errorlimit = 0.01

seeds = [1, 2, 3, 4, 5]
compare_seed = 4

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

def denormalize(y):
    return y * (c1 - c0) + c0

def get_class(y):
    return c0 if abs(y - c0) < abs(y - c1) else c1

def init(seed):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(2, 2) * 0.1
    W2 = rng.randn(2, 1) * 0.1
    b1 = np.zeros((1, 2))
    b2 = np.zeros((1, 1))
    return W1, W2, b1, b2

def forward(x, model, use_relu=False):
    W1, W2, b1, b2 = model

    z = x @ W1 + b1
    h = relu(z) if use_relu else sigmoid(z)
    out = sigmoid(h @ W2 + b2)

    return z, h, out

def train(seed, use_relu=False):
    model = list(init(seed))
    W1, W2, b1, b2 = model
    history = []

    for _ in range(maxepoch):
        error = 0

        for i in range(len(X)):
            x, y = X[i:i+1], Y[i:i+1]
            z, h, out = forward(x, model, use_relu)

            p = np.clip(out, 1e-12, 1 - 1e-12)
            error += (-y * np.log(p) - (1 - y) * np.log(1 - p)).item()

            d_out = out - y
            d_h = d_out @ W2.T
            d_h *= drelu(z) if use_relu else dsigmoid(h)

            W2 -= lr * h.T @ d_out
            b2 -= lr * d_out
            W1 -= lr * x.T @ d_h
            b1 -= lr * d_h

        history.append(error)

        if error <= errorlimit:
            break

    return (W1, W2, b1, b2), history

def predict_norm(a, b, model, use_relu):
    x = np.array([[a / 10, b / 10]])
    return forward(x, model, use_relu)[2][0, 0]

def predict(a, b, model, use_relu):
    return denormalize(predict_norm(a, b, model, use_relu))

def metrics(model, use_relu):
    values = np.array([
        predict(a, b, model, use_relu)
        for a, b, _ in data
    ])

    classes = np.array([get_class(v) for v in values])

    accuracy = np.mean(classes == data[:, 2])
    mae = np.mean(np.abs(values - data[:, 2]))

    return accuracy, mae

sig_runs = []
relu_runs = []

print("Лабораторная работа 3")
print("Вариант 6: c0 = -9, c1 = 6")

print("\nКонфигурация A: Sigmoid")
print("Seed | Эпохи | Ошибка   | Accuracy | MAE")

for seed in seeds:
    model, history = train(seed, False)
    acc, mae = metrics(model, False)

    sig_runs.append((seed, model, history))

    print(
        f"{seed:4} | {len(history):5} | "
        f"{history[-1]:.6f} | {acc:8.2f} | {mae:.4f}"
    )

print("\nКонфигурация B: ReLU")
print("Seed | Эпохи | Ошибка   | Accuracy | MAE")

for seed in seeds:
    model, history = train(seed, True)
    acc, mae = metrics(model, True)

    relu_runs.append((seed, model, history))

    print(
        f"{seed:4} | {len(history):5} | "
        f"{history[-1]:.6f} | {acc:8.2f} | {mae:.4f}"
    )

_, sig_model, sig_history = next(
    x for x in sig_runs if x[0] == compare_seed
)

_, relu_model, relu_history = next(
    x for x in relu_runs if x[0] == compare_seed
)

sig_acc, sig_mae = metrics(sig_model, False)
relu_acc, relu_mae = metrics(relu_model, True)

print(f"\nСравнение, seed={compare_seed}")
print(
    f"{'Метод':<8} | {'Эпохи':>5} | {'Ошибка':>8} | "
    f"{'Accuracy':>8} | {'MAE':>6}"
)
print(
    f"{'Sigmoid':<8} | {len(sig_history):5} | "
    f"{sig_history[-1]:.6f} | {sig_acc:8.2f} | {sig_mae:.4f}"
)
print(
    f"{'ReLU':<8} | {len(relu_history):5} | "
    f"{relu_history[-1]:.6f} | {relu_acc:8.2f} | {relu_mae:.4f}"
)

print("\nПредсказания Sigmoid")

for row in data:
    n = predict_norm(row[0], row[1], sig_model, False)
    y = denormalize(n)

    print(
        f"({row[0]:4.0f}, {row[1]:4.0f}) -> "
        f"norm={n:.4f}, y={y:7.3f}, "
        f"class={get_class(y):3.0f}, expected={row[2]:3.0f}"
    )

print("\nПредсказания ReLU")

for row in data:
    n = predict_norm(row[0], row[1], relu_model, True)
    y = denormalize(n)

    print(
        f"({row[0]:4.0f}, {row[1]:4.0f}) -> "
        f"norm={n:.4f}, y={y:7.3f}, "
        f"class={get_class(y):3.0f}, expected={row[2]:3.0f}"
    )

plt.figure(figsize=(9, 5))
plt.plot(sig_history, label="Sigmoid")
plt.plot(relu_history, label="ReLU")
plt.axhline(errorlimit, linestyle="--", label="Ee = 0.01")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es")
plt.title(f"Сходимость, seed={compare_seed}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

x = np.arange(len(seeds))
width = 0.35

plt.bar(
    x - width / 2,
    [len(h) for _, _, h in sig_runs],
    width,
    label="Sigmoid"
)

for i, (_, _, h) in enumerate(relu_runs):
    plt.bar(
        x[i] + width / 2,
        len(h),
        width,
        label="ReLU" if i == 0 else "",
        hatch="//" if h[-1] > errorlimit else "",
        color="orange"
    )

plt.xlabel("Seed")
plt.ylabel("Количество эпох")
plt.title("Число эпох по 5 запускам")
plt.xticks(x, seeds)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

def plot_surface(model, use_relu, title):
    grid = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(grid, grid)

    Z = np.array([
        [predict(a, b, model, use_relu) for a in grid]
        for b in grid
    ])

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(A, B, Z, levels=30)
    plt.colorbar(contour, label="Выход сети y")

    plt.contour(
        A, B, Z,
        levels=[(c0 + c1) / 2],
        colors="black",
        linewidths=2
    )

    plt.scatter(
        data[data[:, 2] == c0, 0],
        data[data[:, 2] == c0, 1],
        color="blue",
        marker="o",
        s=100,
        label="Класс -9"
    )

    plt.scatter(
        data[data[:, 2] == c1, 0],
        data[data[:, 2] == c1, 1],
        color="red",
        marker="o",
        s=100,
        label="Класс 6"
    )

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_surface(
    sig_model,
    False,
    f"Поверхность Sigmoid, seed={compare_seed}"
)

plot_surface(
    relu_model,
    True,
    f"Поверхность ReLU, seed={compare_seed}"
)

plt.figure(figsize=(7, 5))
plt.bar(["Sigmoid", "ReLU"], [sig_mae, relu_mae])
plt.ylabel("MAE")
plt.title("Сравнение MAE")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nТаблица статистики")
print("Seed | Sigmoid                        | ReLU")
print("     | эпохи  ошибка  Acc   MAE       | эпохи  ошибка  Acc   MAE")

for i in range(len(seeds)):
    s, _, sh = sig_runs[i]
    r, _, rh = relu_runs[i]

    sa, sm = metrics(sig_runs[i][1], False)
    ra, rm = metrics(relu_runs[i][1], True)

    print(
        f"{s:4} | "
        f"{len(sh):5}  {sh[-1]:.4f}  {sa:.2f}  {sm:.3f}     | "
        f"{len(rh):5}  {rh[-1]:.4f}  {ra:.2f}  {rm:.3f}"
    )

print("\nРежим функционирования ReLU")
print("Для выхода: letmeleavepls")

while True:
    s = input("A B [-10 10]: ")

    if s == "letmeleavepls":
        break

    try:
        a, b = map(float, s.split())

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("A и B должны быть в диапазоне [-10; 10]")
            continue

        n = predict_norm(a, b, relu_model, True)
        y = denormalize(n)

        print(f"Нормализованный выход: {n:.4f}")
        print(f"Выход в исходной шкале: {y:.4f}")
        print(f"Класс: {get_class(y):g}")

    except ValueError:
        print("Ошибка ввода. Пример: -3 7")