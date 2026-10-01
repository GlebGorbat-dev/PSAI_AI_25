import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_der(s):
    y = sigmoid(s)
    return y * (1 - y)

def norm(x, c0, c1):
    return (x - c0) / (c1 - c0)

def denorm(y, c0, c1):
    return c0 + y * (c1 - c0)

def mlp_forward(x, W1, T1, W2, T2):
    S1 = np.dot(x, W1) - T1
    h = sigmoid(S1)
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2)
    return S1, h, S2, y

def train_mlp(X, y_true, alpha=0.8, Ee=0.001, max_epochs=10000):
    np.random.seed(42)
    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)
    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1, 1)[0]

    errors = []
    epoch = 0

    while True:
        for xi, yi in zip(X, y_true):
            S1, h, S2, y = mlp_forward(xi, W1, T1, W2, T2)
            
            delta2 = (y - yi) * sigmoid_der(S2)
            delta1 = sigmoid_der(S1) * (W2 * delta2)

            W2 -= alpha * h * delta2
            T2 += alpha * delta2

            W1 -= alpha * np.outer(xi, delta1)
            T1 += alpha * delta1

        Es = 0.0
        for xi, yi in zip(X, y_true):
            _, _, _, y = mlp_forward(xi, W1, T1, W2, T2)
            Es += 0.5 * (y - yi) ** 2
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors

def train_slp(X, y_true, alpha=0.8, Ee=0.001, max_epochs=10000):
    np.random.seed(42)
    W = np.random.uniform(-0.1, 0.1, 2)
    T = np.random.uniform(-0.1, 0.1)

    errors = []
    epoch = 0

    while True:
        for xi, yi in zip(X, y_true):
            S = np.dot(xi, W) - T
            y = sigmoid(S)
            delta = (y - yi) * sigmoid_der(S)

            W -= alpha * xi * delta
            T += alpha * delta

        Es = 0.0
        for xi, yi in zip(X, y_true):
            S = np.dot(xi, W) - T
            y = sigmoid(S)
            Es += 0.5 * (y - yi) ** 2
        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W, T, errors

def main():
    c0 = 1.0
    c1 = 8.0

    X_raw = np.array([
        [1.0, 1.0],
        [1.0, 8.0],
        [8.0, 1.0],
        [8.0, 8.0]
    ])
    y_raw = np.array([1.0, 8.0, 8.0, 1.0])

    X = norm(X_raw, c0, c1)
    y_target = norm(y_raw, c0, c1)

    W1, T1, W2, T2, err_mlp = train_mlp(X, y_target)

    print("--- Результаты многослойного ---")
    print("Эпох:", len(err_mlp))
    print("Ошибка Es:", round(err_mlp[-1], 6))
    
    correct_mlp = 0
    print("Проверка на обучающих примерах:")
    for xi, yi, raw_in, raw_out in zip(X, y_target, X_raw, y_raw):
        _, _, _, pred = mlp_forward(xi, W1, T1, W2, T2)
        real_pred = denorm(pred, c0, c1)
        
        if (pred >= 0.5 and yi >= 0.5) or (pred < 0.5 and yi < 0.5):
            correct_mlp += 1
            
        print("Вход:", raw_in, "Ожидание:", raw_out, "Выход сети:", round(real_pred, 3))
        
    print("Точность:", (correct_mlp / len(X)) * 100, "%")

    W_slp, T_slp, err_slp = train_slp(X, y_target)

    print("\n--- Результаты Однослойного ---")
    print("Эпох:", len(err_slp))
    print("Ошибка Es:", round(err_slp[-1], 6))
    
    correct_slp = 0
    print("Проверка на обучающих примерах:")
    for xi, yi, raw_in, raw_out in zip(X, y_target, X_raw, y_raw):
        S = np.dot(xi, W_slp) - T_slp
        pred = sigmoid(S)
        real_pred = denorm(pred, c0, c1)

        if (pred >= 0.5 and yi >= 0.5) or (pred < 0.5 and yi < 0.5):
            correct_slp += 1

        print("Вход:", raw_in, "Ожидание:", raw_out, "Выход сети:", round(real_pred, 3))

    print("Точность:", (correct_slp / len(X)) * 100, "%")

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(err_mlp, color='blue', linewidth=1.5)
    plt.title("Обучение многослойного", fontsize=11)
    plt.xlabel("Эпохи", fontsize=10)
    plt.ylabel("Ошибка Es", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(err_slp, color='orange', linewidth=1.5)
    plt.title("Обучение Однослойного", fontsize=11)
    plt.xlabel("Эпохи", fontsize=10)
    plt.ylabel("Ошибка Es", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    print("\nРежим проверки (ввод чисел от -10 до 10, 'q' для выхода):")
    while True:
        val = input("Введите A и B: ").strip()
        if val.lower() == 'q':
            break
        items = val.split()
        a = float(items[0])
        b = float(items[1])

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Числа вне диапазона [-10, 10]")
            continue

        x_user = norm(np.array([a, b]), c0, c1)
        _, _, _, y_user = mlp_forward(x_user, W1, T1, W2, T2)
        ans = denorm(y_user, c0, c1)

        c = c0 if abs(ans - c0) < abs(ans - c1) else c1
        print("Выход сети:", round(ans, 4), "| Класс:", int(c))

if __name__ == "__main__":
    main()