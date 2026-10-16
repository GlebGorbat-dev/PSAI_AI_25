import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И ПАРАМЕТРЫ ВАРИАНТА 9
# =====================================================================
CLASS_ZERO = 3.0
CLASS_ONE = -8.0

# Таблица истинности для Варианта 9
DATASET = np.array([
    [ 3.0,  3.0,  3.0],
    [ 3.0, -8.0, -8.0],
    [-8.0,  3.0, -8.0],
    [-8.0, -8.0,  3.0]
], dtype=float)

X_TRAIN = DATASET[:, :2] / 10.0
Y_TRAIN = ((DATASET[:, 2] - CLASS_ZERO) / (CLASS_ONE - CLASS_ZERO)).reshape(-1, 1)

LEARNING_RATE = 0.5
MAX_EPOCHS = 5000
TARGET_ERROR = 0.01

SEEDS_LIST = [1, 2, 3, 4, 5]
MAIN_SEED = 4

# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ И АКТИВАЦИОННЫЕ ФУНКЦИИ
# =====================================================================
def apply_sigmoid(val):
    return 1.0 / (1.0 + np.exp(-np.clip(val, -500, 500)))

def diff_sigmoid(activated_val):
    return activated_val * (1.0 - activated_val)

def apply_relu(val):
    return np.maximum(0.0, val)

def diff_relu(val):
    return np.where(val > 0, 1.0, 0.0)

def restore_scale(norm_val):
    return norm_val * (CLASS_ONE - CLASS_ZERO) + CLASS_ZERO

def find_nearest_class(val):
    dist_0 = np.abs(val - CLASS_ZERO)
    dist_1 = np.abs(val - CLASS_ONE)
    return CLASS_ZERO if dist_0 < dist_1 else CLASS_ONE

# =====================================================================
# АРХИТЕКТУРА И ОБУЧЕНИЕ
# =====================================================================
def create_weights(seed_val):
    generator = np.random.RandomState(seed_val)
    w_hidden = generator.randn(2, 2) * 0.1
    w_output = generator.randn(2, 1) * 0.1
    b_hidden = np.zeros((1, 2))
    b_output = np.zeros((1, 1))
    return {"W1": w_hidden, "W2": w_output, "b1": b_hidden, "b2": b_output}

def pass_forward(inputs, weights, with_relu=False):
    z_layer = inputs @ weights["W1"] + weights["b1"]
    h_layer = apply_relu(z_layer) if with_relu else apply_sigmoid(z_layer)
    out_layer = apply_sigmoid(h_layer @ weights["W2"] + weights["b2"])
    return z_layer, h_layer, out_layer

def fit_network(seed_val, with_relu=False):
    params = create_weights(seed_val)
    err_history = []

    for epoch in range(MAX_EPOCHS):
        total_loss = 0.0

        for idx in range(len(X_TRAIN)):
            x_sample = X_TRAIN[idx:idx+1]
            y_sample = Y_TRAIN[idx:idx+1]

            z1, h1, pred = pass_forward(x_sample, params, with_relu)

            p_clipped = np.clip(pred, 1e-12, 1.0 - 1e-12)
            total_loss += float(-y_sample * np.log(p_clipped) - (1.0 - y_sample) * np.log(1.0 - p_clipped))

            grad_out = pred - y_sample
            grad_hidden = grad_out @ params["W2"].T
            grad_hidden *= diff_relu(z1) if with_relu else diff_sigmoid(h1)

            params["W2"] -= LEARNING_RATE * (h1.T @ grad_out)
            params["b2"] -= LEARNING_RATE * grad_out
            params["W1"] -= LEARNING_RATE * (x_sample.T @ grad_hidden)
            params["b1"] -= LEARNING_RATE * grad_hidden

        err_history.append(total_loss)

        if total_loss <= TARGET_ERROR:
            break

    return params, err_history

# =====================================================================
# ОЦЕНКА И МЕТРИКИ
# =====================================================================
def eval_norm_output(inp_a, inp_b, params, with_relu):
    vec = np.array([[inp_a / 10.0, inp_b / 10.0]])
    return pass_forward(vec, params, with_relu)[2][0, 0]

def eval_scaled_output(inp_a, inp_b, params, with_relu):
    norm_res = eval_norm_output(inp_a, inp_b, params, with_relu)
    return restore_scale(norm_res)

def compute_performance(params, with_relu):
    preds = [
        eval_scaled_output(row[0], row[1], params, with_relu)
        for row in DATASET
    ]
    preds = np.array(preds)
    assigned_classes = np.array([find_nearest_class(v) for v in preds])

    acc = np.mean(assigned_classes == DATASET[:, 2])
    mae_err = np.mean(np.abs(preds - DATASET[:, 2]))

    return acc, mae_err

# =====================================================================
# ОСНОВНОЙ СЦЕНАРИЙ И ВЫВОД
# =====================================================================
print("Лабораторная работа 3")
print("Вариант 9: c0 = 3, c1 = -8")

sigmoid_results = []
relu_results = []

print("\nКонфигурация A: Sigmoid")
print("Seed | Эпохи | Ошибка    | Accuracy | MAE")

for s in SEEDS_LIST:
    net_params, hist = fit_network(s, with_relu=False)
    accuracy, mae = compute_performance(net_params, with_relu=False)
    sigmoid_results.append((s, net_params, hist))
    print(f"{s:4} | {len(hist):5} | {hist[-1]:.6f} | {accuracy:8.2f} | {mae:.4f}")

print("\nКонфигурация B: ReLU")
print("Seed | Эпохи | Ошибка    | Accuracy | MAE")

for s in SEEDS_LIST:
    net_params, hist = fit_network(s, with_relu=True)
    accuracy, mae = compute_performance(net_params, with_relu=True)
    relu_results.append((s, net_params, hist))
    print(f"{s:4} | {len(hist):5} | {hist[-1]:.6f} | {accuracy:8.2f} | {mae:.4f}")

# Сравнение по фиксированному seed
_, best_sig_params, best_sig_hist = next(item for item in sigmoid_results if item[0] == MAIN_SEED)
_, best_relu_params, best_relu_hist = next(item for item in relu_results if item[0] == MAIN_SEED)

acc_sig_main, mae_sig_main = compute_performance(best_sig_params, with_relu=False)
acc_relu_main, mae_relu_main = compute_performance(best_relu_params, with_relu=True)

print(f"\nСравнение, seed={MAIN_SEED}")
print(f"{'Метод':<8} | {'Эпохи':>5} | {'Ошибка':>8} | {'Accuracy':>8} | {'MAE':>6}")
print(f"{'Sigmoid':<8} | {len(best_sig_hist):5} | {best_sig_hist[-1]:.6f} | {acc_sig_main:8.2f} | {mae_sig_main:.4f}")
print(f"{'ReLU':<8} | {len(best_relu_hist):5} | {best_relu_hist[-1]:.6f} | {acc_relu_main:8.2f} | {mae_relu_main:.4f}")

print("\nПредсказания Sigmoid")
for sample in DATASET:
    n_out = eval_norm_output(sample[0], sample[1], best_sig_params, with_relu=False)
    s_out = restore_scale(n_out)
    print(f"({sample[0]:4.0f}, {sample[1]:4.0f}) -> norm={n_out:.4f}, y={s_out:7.3f}, class={find_nearest_class(s_out):3.0f}, expected={sample[2]:3.0f}")

print("\nПредсказания ReLU")
for sample in DATASET:
    n_out = eval_norm_output(sample[0], sample[1], best_relu_params, with_relu=True)
    s_out = restore_scale(n_out)
    print(f"({sample[0]:4.0f}, {sample[1]:4.0f}) -> norm={n_out:.4f}, y={s_out:7.3f}, class={find_nearest_class(s_out):3.0f}, expected={sample[2]:3.0f}")

# =====================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =====================================================================
# 1. График сходимости
plt.figure(figsize=(9, 5))
plt.plot(best_sig_hist, label="Sigmoid")
plt.plot(best_relu_hist, label="ReLU")
plt.axhline(TARGET_ERROR, linestyle="--", label="Ee = 0.01")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es")
plt.title(f"Сходимость, seed={MAIN_SEED}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Сравнение количества эпох по запускам
plt.figure(figsize=(9, 5))
x_positions = np.arange(len(SEEDS_LIST))
bar_width = 0.35

plt.bar(x_positions - bar_width / 2, [len(h) for _, _, h in sigmoid_results], bar_width, label="Sigmoid")

for i, (_, _, h) in enumerate(relu_results):
    plt.bar(
        x_positions[i] + bar_width / 2,
        len(h),
        bar_width,
        label="ReLU" if i == 0 else "",
        hatch="//" if h[-1] > TARGET_ERROR else "",
        color="orange"
    )

plt.xlabel("Seed")
plt.ylabel("Количество эпох")
plt.title("Число эпох по 5 запускам")
plt.xticks(x_positions, SEEDS_LIST)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# 3. Карты разделяющих поверхностей
def draw_decision_surface(params, with_relu, plot_title):
    ticks = np.linspace(-10, 10, 200)
    grid_x, grid_y = np.meshgrid(ticks, ticks)

    surface_z = np.array([
        [eval_scaled_output(ax, bx, params, with_relu) for ax in ticks]
        for bx in ticks
    ])

    plt.figure(figsize=(8, 6))
    contour_fill = plt.contourf(grid_x, grid_y, surface_z, levels=30)
    plt.colorbar(contour_fill, label="Выход сети y")

    plt.contour(
        grid_x, grid_y, surface_z,
        levels=[(CLASS_ZERO + CLASS_ONE) / 2.0],
        colors="black", linewidths=2
    )

    plt.scatter(
        DATASET[DATASET[:, 2] == CLASS_ZERO, 0],
        DATASET[DATASET[:, 2] == CLASS_ZERO, 1],
        color="blue", marker="o", s=100, label="Класс 3"
    )
    plt.scatter(
        DATASET[DATASET[:, 2] == CLASS_ONE, 0],
        DATASET[DATASET[:, 2] == CLASS_ONE, 1],
        color="red", marker="o", s=100, label="Класс -8"
    )

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(plot_title)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

draw_decision_surface(best_sig_params, False, f"Поверхность Sigmoid, seed={MAIN_SEED}")
draw_decision_surface(best_relu_params, True, f"Поверхность ReLU, seed={MAIN_SEED}")

# 4. Диаграмма сравнения MAE
plt.figure(figsize=(7, 5))
plt.bar(["Sigmoid", "ReLU"], [mae_sig_main, mae_relu_main])
plt.ylabel("MAE")
plt.title("Сравнение MAE")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# =====================================================================
# ТАБЛИЦА СТАТИСТИКИ И ИНТЕРАКТИВНЫЙ РЕЖИМ
# =====================================================================
print("\nТаблица статистики")
print("Seed | Sigmoid                         | ReLU")
print("     | эпохи  ошибка  Acc   MAE        | эпохи  ошибка  Acc   MAE")

for i in range(len(SEEDS_LIST)):
    s_val, _, s_h = sigmoid_results[i]
    _, _, r_h = relu_results[i]

    s_acc, s_mae = compute_performance(sigmoid_results[i][1], with_relu=False)
    r_acc, r_mae = compute_performance(relu_results[i][1], with_relu=True)

    print(
        f"{s_val:4} | "
        f"{len(s_h):5}  {s_h[-1]:.4f}  {s_acc:.2f}  {s_mae:.3f}     | "
        f"{len(r_h):5}  {r_h[-1]:.4f}  {r_acc:.2f}  {r_mae:.3f}"
    )

print("\nРежим функционирования ReLU")
print("Для выхода: letmeleavepls")

while True:
    user_inp = input("A B [-10 10]: ")

    if user_inp.strip() == "letmeleavepls":
        break

    try:
        val_a, val_b = map(float, user_inp.split())

        if not (-10.0 <= val_a <= 10.0 and -10.0 <= val_b <= 10.0):
            print("A и B должны быть в диапазоне [-10; 10]")
            continue

        n_val = eval_norm_output(val_a, val_b, best_relu_params, with_relu=True)
        s_val = restore_scale(n_val)

        print(f"Нормализованный выход: {n_val:.4f}")
        print(f"Выход в исходной шкале: {s_val:.4f}")
        print(f"Класс: {find_nearest_class(s_val):g}")

    except ValueError:
        print("Ошибка ввода. Пример: -3 7")