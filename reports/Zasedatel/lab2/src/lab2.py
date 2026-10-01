import numpy as np
import matplotlib.pyplot as plt

c0, c1 = -9, 6

data = np.array([
    [-9, -9, -9],
    [-9,  6,  6],
    [ 6, -9,  6],
    [ 6,  6, -9]
], dtype=float)

X = data[:, :2] / 10  #
Y = ((data[:, 2] - c0) / (c1 - c0)).reshape(-1, 1)  

lr = 0.5
maxepoch = 10000
errorlimit = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def denormalize(y):
    return y * (c1 - c0) + c0

def get_class(y):
    return c0 if abs(y - c0) < abs(y - c1) else c1

def init(seed):  # веса задаются функцией с параметром seed — нужно для серии из 5 запусков (в ЛР1 seed был один, фиксированный)
    rng = np.random.RandomState(seed)
    W1 = rng.randn(2, 2) * 0.1
    W2 = rng.randn(2, 1) * 0.1
    b1 = np.zeros((1, 2))
    b2 = np.zeros((1, 1))
    return W1, W2, b1, b2

def forward(x, W1, W2, b1, b2):  
    h = sigmoid(x @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    return h, out

def train_mse(seed):  # то же обучение, что в ЛР1 (MSE), но обёрнуто в функцию под конкретный seed
    W1, W2, b1, b2 = init(seed)
    history = []

    for epoch in range(maxepoch):
        error = 0

        for i in range(len(X)):
            x, y = X[i:i+1], Y[i:i+1]
            h, out = forward(x, W1, W2, b1, b2)

            error += np.sum((y - out) ** 2)  

            d_out = (y - out) * dsigmoid(out)  
            d_h = d_out @ W2.T * dsigmoid(h)

            W2 += lr * h.T @ d_out
            b2 += lr * d_out
            W1 += lr * x.T @ d_h
            b1 += lr * d_h

        history.append(error)  # сохраняем ошибку каждой эпохи для графика сходимости

        if error <= errorlimit:
            break

    return W1, W2, b1, b2, history

def train_bce(seed):  # вторая функция потерь — BCE вместо MSE, структура цикла та же
    W1, W2, b1, b2 = init(seed)
    history = []

    for epoch in range(maxepoch):
        error = 0

        for i in range(len(X)):
            x, y = X[i:i+1], Y[i:i+1]
            h, out = forward(x, W1, W2, b1, b2)

            p = np.clip(out, 1e-12, 1 - 1e-12)  # защита от log(0)
            error += np.sum(
                -y * np.log(p) - (1 - y) * np.log(1 - p)  # формула BCE вместо (y-out)**2
            )

            d_out = out - y  # градиент BCE+сигмоида = out - y 
            d_h = d_out @ W2.T * dsigmoid(h)  # для скрытого слоя правило то же, что в ЛР1

            W2 -= lr * h.T @ d_out  
            b2 -= lr * d_out
            W1 -= lr * x.T @ d_h
            b1 -= lr * d_h

        history.append(error)

        if error <= errorlimit:
            break

    return W1, W2, b1, b2, history

def predict_norm(a, b, model):  # возвращает "сырой" нормализованный выход y∈(0;1) без денормализации
    W1, W2, b1, b2 = model
    x = np.array([[a / 10, b / 10]])
    _, out = forward(x, W1, W2, b1, b2)
    return out[0, 0]

def predict(a, b, model):  # аналог predict() из ЛР1, но принимает конкретную модель (нужно для сравнения MSE/BCE)
    return denormalize(predict_norm(a, b, model))

def metrics(model):  # считает accuracy и MAE одной функцией — нужно для серии из 5 запусков и сравнения конфигураций
    values = np.array([
        predict(row[0], row[1], model)
        for row in data
    ])

    classes = np.array([
        get_class(v)
        for v in values
    ])

    accuracy = np.mean(classes == data[:, 2])
    mae = np.mean(np.abs(values - data[:, 2]))  # средняя абсолютная ошибка в шкале [c0;c1] (в ЛР1 не считалась)

    return accuracy, mae

seeds = [1, 2, 3, 4, 5]  # серия из 5 seed вместо одного фиксированного, как требует задание ЛР2
compare_seed = 4

mse_runs = []
bce_runs = []

print("Лабораторная номер 2")
print("Вариант 6: c0 = -9, c1 = 6")

print("\nЗапуски MSE")

for seed in seeds:  # цикл по seed вместо однократного обучения (сбор статистики устойчивости)
    *model, history = train_mse(seed)
    accuracy, mae = metrics(model)

    mse_runs.append((seed, model, history))

    print(
        f"seed={seed}: эпох={len(history)}, "
        f"ошибка={history[-1]:.6f}, "
        f"accuracy={accuracy:.2f}, MAE={mae:.4f}"
    )

print("\nЗапуски BCE")

for seed in seeds:
    *model, history = train_bce(seed)
    accuracy, mae = metrics(model)

    bce_runs.append((seed, model, history))

    print(
        f"seed={seed}: эпох={len(history)}, "
        f"ошибка={history[-1]:.6f}, "
        f"accuracy={accuracy:.2f}, MAE={mae:.4f}"
    )

mse_seed, mse_model, mse_history = next(
    x for x in mse_runs if x[0] == compare_seed
)  # выбор одного seed для прямого сравнения MSE и BCE на одинаковой инициализации

bce_seed, bce_model, bce_history = next(
    x for x in bce_runs if x[0] == compare_seed
)

mse_acc, mse_mae = metrics(mse_model)
bce_acc, bce_mae = metrics(bce_model)

print("Сравнение MSE и BCE")
print("Один и тот же seed:", compare_seed)

print("\nMSE:")
print("Эпохи:", len(mse_history))
print("Ошибка:", round(mse_history[-1], 6))
print("Accuracy:", mse_acc)
print("MAE:", round(mse_mae, 6))

print("\nBCE:")
print("Эпохи:", len(bce_history))
print("Ошибка:", round(bce_history[-1], 6))
print("Accuracy:", bce_acc)
print("MAE:", round(bce_mae, 6))

print("Предсказания MSE")

for row in data:
    norm = predict_norm(row[0], row[1], mse_model)
    value = denormalize(norm)

    print(
        f"({row[0]:2.0f},{row[1]:2.0f}) | "
        f"y_norm={norm:.4f} | "  # вывод нормализованного значения y (в ЛР1 показывался только пересчитанный y)
        f"y={value:7.3f} | "
        f"class={get_class(value):2.0f} | "
        f"expected={row[2]:2.0f}"
    )

print("Предсказания BCE")

for row in data:
    norm = predict_norm(row[0], row[1], bce_model)
    value = denormalize(norm)

    print(
        f"({row[0]:2.0f},{row[1]:2.0f}) | "
        f"y_norm={norm:.4f} | "
        f"y={value:7.3f} | "
        f"class={get_class(value):2.0f} | "
        f"expected={row[2]:2.0f}"
    )

# весь блок графиков ниже отсутствовал в ЛР1 — требование задания ЛР2 (визуализация сходимости, устойчивости, границы, точности)

plt.figure(figsize=(10, 5))
plt.plot(mse_history, label="MSE")
plt.plot(bce_history, label="BCE")
plt.axhline(errorlimit, linestyle="--", label="Ee = 0.01")
plt.xlabel("Номер эпохи")
plt.ylabel("Суммарная ошибка")
plt.title(f"Сходимость MSE и BCE, seed={compare_seed}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

x = np.arange(len(seeds))
width = 0.35

mse_epochs = [len(h) for _, _, h in mse_runs]
bce_epochs = [len(h) for _, _, h in bce_runs]

mse_hatch = [
    "" if h[-1] <= errorlimit else "//"  # штриховка запусков, не достигших критерия остановки
    for _, _, h in mse_runs
]

bce_hatch = [
    "" if h[-1] <= errorlimit else "//"
    for _, _, h in bce_runs
]

for i in range(len(seeds)):
    plt.bar(
        x[i] - width / 2,
        mse_epochs[i],
        width,
        label="MSE" if i == 0 else "",
        hatch=mse_hatch[i],
        color="steelblue"
    )

for i in range(len(seeds)):
    plt.bar(
        x[i] + width / 2,
        bce_epochs[i],
        width,
        label="BCE" if i == 0 else "",
        hatch=bce_hatch[i],
        color="orange"
    )

plt.xlabel("Seed")
plt.ylabel("Количество эпох")
plt.title("Сравнение числа эпох по 5 запускам")
plt.xticks(x, seeds)

plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

def plot_boundary(model, title):  # визуализация разделяющей поверхности на плоскости (A,B) heatmap + граница
    grid = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(grid, grid)

    Z = np.array([
        [predict(a, b, model) for a in grid]
        for b in grid
    ])

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(
        A, B, Z,
        levels=30,
        alpha=0.8
    )

    plt.colorbar(
        contour,
        label="Выход сети y"
    )

    plt.contour(
        A, B, Z,
        levels=[(c0 + c1) / 2],  # линия уровня посередине между классами
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

plot_boundary(
    mse_model,
    f"Разделяющая граница MSE, seed={compare_seed}"
)

plot_boundary(
    bce_model,
    f"Разделяющая граница BCE, seed={compare_seed}"
)

plt.figure(figsize=(7, 5))  # диаграмма сравнения MAE между конфигурациями
plt.bar(["MSE", "BCE"], [mse_mae, bce_mae])
plt.ylabel("Средняя абсолютная ошибка")
plt.title("Сравнение точности восстановления исходной шкалы")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("Режим функционирования BCE")
print("Введите A B из диапазона [-10; 10]")
print("Для выхода: letmeleavepls")

while True:  # то же, что в ЛР1, но с проверкой диапазона и выводом обоих значений (норм. и пересчитанного)
    s = input("A B [-10 10]: ")

    if s == "letmeleavepls":
        break

    try:
        a, b = map(float, s.split())

        if not (-10 <= a <= 10 and -10 <= b <= 10):  # явная проверка диапазона (в ЛР1 отсутствовала)
            print("A и B должны быть в диапазоне [-10; 10]")
            continue

        norm = predict_norm(a, b, bce_model)
        value = denormalize(norm)

        print(f"Нормализованный выход: {norm:.4f}")  # вывод y_norm отдельно (в ЛР1 сразу выводился y)
        print(f"Выход в исходной шкале: {value:.4f}")
        print(f"Ближайший класс: {get_class(value):g}")

    except ValueError:  # обработка ошибок ввода
        print("Ошибка ввода. Пример: -3 7")