import numpy as np
import matplotlib.pyplot as plt

def activation(S):
    return 1 / (1 + np.exp(-S))

def activation_derivative(S):
    y = activation(S)
    return y * (1 - y)

def normalize(value, c0, c1):
    return (value - c0) / (c1 - c0)

def denormalize(value, c0, c1):
    return c0 + value * (c1 - c0)

def forward(x, W1, T1, W2, T2):
    S1 = np.dot(x, W1) - T1
    h = activation(S1)

    S2 = np.dot(h, W2) - T2
    y = activation(S2)

    return S1, h, S2, y

def predict(x, W1, T1, W2, T2):
    _, _, _, y = forward(x, W1, T1, W2, T2)
    return y

def online_train(X, e, alpha=0.1, Ee=0.01,  max_epochs=200):

    np.random.seed(42)
    W1 = np.random.uniform(-0.1, 0.1, (2, 2))
    T1 = np.random.uniform(-0.1, 0.1, 2)
    W2 = np.random.uniform(-0.1, 0.1, 2)
    T2 = np.random.uniform(-0.1, 0.1, 1)[0]

    errors = []
    epoch = 0

    while True:
        for xi, ei in zip(X, e):

            S1, h, S2, y = forward(
                xi, W1, T1, W2, T2
            )

            delta2 = (y - ei) * activation_derivative(S2)

            delta1 = activation_derivative(S1) * W2 * delta2

            W2 = W2 - alpha * h * delta2
            T2 = T2 + alpha * delta2

            W1 = W1 - alpha * np.outer(xi, delta1)
            T1 = T1 + alpha * delta1

        Es = 0
        for xi, ei in zip(X, e):
            y = predict(xi, W1, T1, W2, T2)
            Es += 0.5 * (y - ei) ** 2
        errors.append(Es)
        epoch += 1
        
        if Es <= Ee or epoch >= max_epochs:
            break

    return W1, T1, W2, T2, errors

def slp_train(X, e, alpha=0.9, Ee=0.001, max_epochs=10000):

    np.random.seed(42)

    W = np.random.uniform(-0.1, 0.1, 2)
    T = np.random.uniform(-0.1, 0.1)

    errors = []
    epoch = 0

    while True:

        for xi, ei in zip(X, e):

            S = np.dot(xi, W) - T
            y = activation(S)

            delta = (y - ei) * activation_derivative(S)

            W = W - alpha * xi * delta
            T = T + alpha * delta

        Es = 0

        for xi, ei in zip(X, e):

            S = np.dot(xi, W) - T
            y = activation(S)

            Es += 0.5 * (y - ei) ** 2

        errors.append(Es)
        epoch += 1

        if Es <= Ee or epoch >= max_epochs:
            break

    return W, T, errors
    
def show_plot(errors):
    plt.figure()

    plt.plot(
        errors,
        label="Online learning"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("График ошибки обучения")
    plt.legend()
    plt.grid()

    plt.show()


def main():
    c0, c1 = 4, -1

    x1 = np.array([4, 4, -1, -1])
    x2 = np.array([4, -1, 4, -1])
    e = np.array([4, -1, -1, 4])

    X = np.vstack([x1, x2]).T

    X = normalize(X, c0, c1)
    e = normalize(e, c0, c1)

    W1, T1, W2, T2, errors = online_train(X, e, alpha=0.9, Ee=0.001, max_epochs=10000)

    print("\nМногослойный персептрон:")
    print("Эпох обучения:", len(errors))
    print("Финальная ошибка:", errors[-1])
    print("Ответы сети:")

    for xi, ei in zip(X, e):
        y = predict(xi, W1, T1, W2, T2)

        print(
            f"Вход: {denormalize(xi, c0, c1)}, "
            f"Ожидалось: {denormalize(ei, c0, c1)}, "
            f"Получено: {denormalize(y, c0, c1)}"
        )

    show_plot(errors)

    W_p, T_p, errors_p = slp_train(X, e, alpha=0.9, Ee=0.001, max_epochs=10000)
    print("\nОднослойный персептрон:")
    print("Эпох обучения:", len(errors_p))
    print("Финальная ошибка:", errors_p[-1])

    print("Ответы персептрона:")

    for xi, ei in zip(X, e):

        S = np.dot(xi, W_p) - T_p
        y = activation(S)

        print(
            f"Вход: {denormalize(xi, c0, c1)}, "
            f"Ожидалось: {denormalize(ei, c0, c1)}, "
            f"Получено: {denormalize(y, c0, c1)}"
        )


    show_plot(errors_p)

    while (True):
        user_input = input(
            "Введите пару чисел (A, B) из диапазона [-10; 10]: "
        )

        if user_input.lower() == "q":
            break

        a, b = map(float, user_input.split())

        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Числа должны быть в диапазоне [-10; 10]")
            continue

        x = normalize(np.array([a, b]), c0, c1)

        y = predict(x, W1, T1, W2, T2)

        y_real = denormalize(y, c0, c1)

        if abs(y_real - c0) < abs(y_real - c1):
            class_result = c0
        else:
            class_result = c1

        print(f"Выход сети: {y_real:.4f}")
        print(f"Ближайший класс: {class_result}")


if __name__ == "__main__":
    main()