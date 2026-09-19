import numpy as np
import matplotlib.pyplot as plt

c0, c1 = 3.0, -8.0

def sigmoid(S):
    return 1.0 / (1.0 + np.exp(-S))

def sigmoid_deriv(y):
    return y * (1.0 - y)

X_raw = np.array([[3, 3], [3, -8], [-8, 3], [-8, -8]], dtype=float)
E_raw = np.array([3, -8, -8, 3], dtype=float)

X = (X_raw + 8) / 11
E = (E_raw + 8) / 11

print("Нормализованные входы:")
print(X)
print("Нормализованные эталоны:", E)

np.random.seed(42)
W1 = np.random.uniform(-0.1, 0.1, (2, 2))
T1 = np.random.uniform(-0.1, 0.1, 2)
W2 = np.random.uniform(-0.1, 0.1, 2)
T2 = float(np.random.uniform(-0.1, 0.1))

alpha = 0.5
beta = 0.9
Ee = 0.005
max_epochs = 20000
mlp_errors = []

dW1_prev = np.zeros_like(W1)
dT1_prev = np.zeros_like(T1)
dW2_prev = np.zeros_like(W2)
dT2_prev = 0.0

for epoch in range(max_epochs):
    Es = 0.0
    for k in range(4):
        x = X[k]
        e = E[k]

        S1 = W1 @ x - T1
        y1 = sigmoid(S1)
        S2 = float(W2 @ y1 - T2)
        y3 = float(sigmoid(S2))

        gamma3 = y3 - e
        delta3 = gamma3 * sigmoid_deriv(y3)
        gamma1 = delta3 * W2 * sigmoid_deriv(y1)
        delta1 = gamma1

        dW2 = -alpha * delta3 * y1 + beta * dW2_prev
        dT2 = alpha * delta3 + beta * dT2_prev
        dW1 = -alpha * np.outer(delta1, x) + beta * dW1_prev
        dT1 = alpha * delta1 + beta * dT1_prev

        W2 += dW2
        T2 += dT2
        W1 += dW1
        T1 += dT1

        dW1_prev = dW1
        dT1_prev = dT1
        dW2_prev = dW2
        dT2_prev = dT2

        Es += 0.5 * (y3 - e) ** 2

    mlp_errors.append(Es)

    if epoch % 1000 == 0:
        print("MLP эпоха", epoch, "| Es =", round(Es, 8))

    if Es <= Ee:
        print()
        print(">>> MLP: сходимость за", epoch + 1, "эпох, Es =", round(Es, 8))
        break
else:
    print()
    print(">>> MLP: лимит", max_epochs, "эпох, Es =", round(Es, 8))

print()
print("--- Результаты MLP ---")
correct_mlp = 0
for k in range(4):
    x = X[k]
    e_raw = E_raw[k]
    S1 = W1 @ x - T1
    y1 = sigmoid(S1)
    S2 = float(W2 @ y1 - T2)
    y3 = float(sigmoid(S2))
    y_raw = y3 * 11 - 8
    pred = c0 if abs(y_raw - c0) < abs(y_raw - c1) else c1
    ok = (pred == e_raw)
    correct_mlp += int(ok)
    print("A =", int(X_raw[k][0]), "| B =", int(X_raw[k][1]),
          "| эталон =", int(e_raw), "| выход =", round(y_raw, 3),
          "| класс =", int(pred), "|", "OK" if ok else "FAIL")

print("Точность MLP:", correct_mlp, "/ 4 =", correct_mlp / 4 * 100, "%")

np.random.seed(42)
w1 = float(np.random.uniform(-0.1, 0.1))
w2 = float(np.random.uniform(-0.1, 0.1))
T_single = float(np.random.uniform(-0.1, 0.1))

slp_errors = []

for epoch in range(max_epochs):
    Es = 0.0
    for k in range(4):
        x = X[k]
        e = E[k]
        S = float(w1 * x[0] + w2 * x[1] - T_single)
        y = float(sigmoid(S))

        gamma = y - e
        delta = gamma * sigmoid_deriv(y)

        w1 -= alpha * delta * x[0]
        w2 -= alpha * delta * x[1]
        T_single += alpha * delta

        Es += 0.5 * (y - e) ** 2

    slp_errors.append(Es)

    if epoch % 1000 == 0:
        print("SLP эпоха", epoch, "| Es =", round(Es, 8))

    if Es <= Ee:
        print()
        print(">>> SLP: сходимость за", epoch + 1, "эпох, Es =", round(Es, 8))
        break
else:
    print()
    print(">>> SLP: НЕ сошёлся за", max_epochs, "эпох, Es =", round(Es, 8))

print()
print("--- Результаты однослойного персептрона ---")
correct_slp = 0
for k in range(4):
    x = X[k]
    e_raw = E_raw[k]
    S = float(w1 * x[0] + w2 * x[1] - T_single)
    y = float(sigmoid(S))
    y_raw = y * 11 - 8
    pred = c0 if abs(y_raw - c0) < abs(y_raw - c1) else c1
    ok = (pred == e_raw)
    correct_slp += int(ok)
    print("A =", int(X_raw[k][0]), "| B =", int(X_raw[k][1]),
          "| эталон =", int(e_raw), "| выход =", round(y_raw, 3),
          "| класс =", int(pred), "|", "OK" if ok else "FAIL")

print("Точность SLP:", correct_slp, "/ 4 =", correct_slp / 4 * 100, "%")


print()
print("=" * 60)
print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ ОБУЧЕННОЙ СЕТИ")
print("=" * 60)
print("Введите пару чисел (A, B) из диапазона [-10; 10].")
print("Для выхода введите 'q'.")
print()

while True:
    user_input = input("Введите A и B через пробел (или 'q' для выхода): ").strip()

    if user_input.lower() == 'q':
        print("Выход из режима функционирования.")
        break

    parts = user_input.split()

    if len(parts) != 2:
        print("Ошибка: введите ровно два числа через пробел.")
        continue

    try:
        A = float(parts[0])
        B = float(parts[1])
    except ValueError:
        print("Ошибка: введите числа.")
        continue

    if A < -10 or A > 10 or B < -10 or B > 10:
        print("Ошибка: числа должны быть в диапазоне [-10; 10].")
        continue


    a_norm = (A + 8) / 11
    b_norm = (B + 8) / 11


    x = np.array([a_norm, b_norm])
    S1 = W1 @ x - T1
    y1 = sigmoid(S1)
    S2 = float(W2 @ y1 - T2)
    y3 = float(sigmoid(S2))


    y_hat = y3 * 11 - 8


    if abs(y_hat - c0) < abs(y_hat - c1):
        pred_class = c0
        class_name = "c0 (логический 0)"
    else:
        pred_class = c1
        class_name = "c1 (логическая 1)"


    print("A =", A, "| B =", B)
    print("Выход сети: y_hat =", round(y_hat, 4))
    print("Класс:", int(pred_class), "-", class_name)
    print("Расстояние до c0 =", round(abs(y_hat - c0), 4))
    print("Расстояние до c1 =", round(abs(y_hat - c1), 4))
    print("-" * 50)
