import numpy as np
import matplotlib.pyplot as plt

# параметры варианта 6
a, b, c, d = 1.2, 0.28, 0.6, 0.65
p_var, N_var = 6, 4
K = 200
TRAIN_END = 140       # k=0...139 - train, k=140...199 - test

LR = 0.1               # шаг обучения, взят из диапазона [0.01; 0.3]
MAXEPOCH = 20000
ERRORLIMIT = 1e-4

def gen_series():
    # генерируем ряд по формуле варианта
    k = np.arange(K)
    x = a * np.cos(b * k) + c * np.sin(d * k)
    return k, x

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(h):
    return h * (1 - h)

def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def denormalize(x_norm, xmin, xmax):
    return x_norm * (xmax - xmin) + xmin

def make_windows(x_norm, p, k_start, k_end):
    # вход - p предыдущих значений, цель - следующее значение
    X, Y, ks = [], [], []
    for k in range(k_start, k_end + 1):
        X.append(x_norm[k - p:k])
        Y.append(x_norm[k])
        ks.append(k)
    return np.array(X), np.array(Y).reshape(-1, 1), np.array(ks)

def init_weights(seed, p, N):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(p, N) * 0.1
    W2 = rng.randn(N, 1) * 0.1
    b1 = np.zeros((1, N))
    b2 = np.zeros((1, 1))
    return [W1, W2, b1, b2]

def forward(x, model):
    W1, W2, b1, b2 = model
    h = sigmoid(x @ W1 + b1)
    out = h @ W2 + b2      # выход линейный, f(s)=s
    return h, out

def train(Xtr, Ytr, seed, p, N, maxep=MAXEPOCH, lr=LR, errorlimit=ERRORLIMIT):
    model = init_weights(seed, p, N)
    W1, W2, b1, b2 = model
    history = []
    n = len(Xtr)
    rng = np.random.RandomState(seed + 10000)   # свой генератор для перемешивания

    for epoch in range(maxep):
        order = rng.permutation(n)   # перемешиваем порядок примеров каждую эпоху
        error = 0.0

        for i in order:
            x, y = Xtr[i:i + 1], Ytr[i:i + 1]
            h, out = forward(x, model)

            error += np.sum((y - out) ** 2)

            # градиент: линейный выход -> (y-out), сигм. скрытый -> умножаем на h(1-h)
            d_out = (y - out)
            d_h = d_out @ W2.T * dsigmoid(h)

            W2 += lr * h.T @ d_out
            b2 += lr * d_out
            W1 += lr * x.T @ d_h
            b1 += lr * d_h

        error /= n   # средняя ошибка по выборке
        history.append(error)

        if error <= errorlimit:
            break

    return model, history

def predict_norm(x_window, model):
    _, out = forward(x_window.reshape(1, -1), model)
    return out[0, 0]

def metrics(model, X, Y, xmin, xmax):
    # считаем метрики в исходной шкале
    preds_norm = np.array([predict_norm(X[i], model) for i in range(len(X))])
    preds = denormalize(preds_norm, xmin, xmax)
    true = denormalize(Y.ravel(), xmin, xmax)

    err = preds - true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    # ряд колеблется около нуля, поэтому берём sMAPE вместо MAPE
    denom = np.abs(true) + np.abs(preds)
    denom[denom == 0] = 1e-9
    smape = np.mean(2 * np.abs(err) / denom) * 100

    return mae, rmse, smape, preds, true

def recursive_forecast(model, x_norm_full, p, start_k, steps):
    # рекурсивный прогноз: каждое предсказание подставляется в окно как вход
    window = x_norm_full[start_k - p:start_k].copy()
    preds_norm = []
    for _ in range(steps):
        yhat = predict_norm(window, model)
        preds_norm.append(yhat)
        window = np.append(window[1:], yhat)
    return np.array(preds_norm)


def fmt_window(window_norm):
    # ровный однострочный вывод окна без переносов numpy
    return "[" + ", ".join(f"{v:.4f}" for v in window_norm) + "]"


def show_prediction(window_norm, model, xmin, xmax, true_val=None):
    # печатает один пример в едином формате: вход, прогноз норм., прогноз в шкале ряда, ошибка
    pn = predict_norm(window_norm, model)
    pred = denormalize(pn, xmin, xmax)

    print(f"    вход (норм.):     {fmt_window(window_norm)}")
    print(f"    прогноз (норм.):  {pn:.4f}")
    print(f"    прогноз (ряд):    {pred:.4f}")
    if true_val is not None:
        print(f"    истинное x(k):    {true_val:.4f}")
        print(f"    абс. ошибка:      {abs(pred - true_val):.4f}")
    print()
    return pred

def functioning_mode(model, p, x, x_norm, xmin, xmax, train_end):
    # интерактивный режим: пользователь вводит либо k, либо p значений ряда
    print(f"\nРежим функционирования (сеть p={p})")
    print(f"Введите номер k ({p}..{K-1}) чтобы взять окно из ряда,")
    print(f"или {p} чисел через пробел - своё окно в исходной шкале.")
    print("Для выхода: letmeleavepls")

    while True:
        s = input("> ").strip()

        if s == "letmeleavepls":
            break

        parts = s.split()

        if len(parts) == 1:
            try:
                kk = int(parts[0])
            except ValueError:
                print("Ошибка: ожидался номер k"); continue

            if kk < p or kk >= K:
                print(f"k должен быть в диапазоне [{p}; {K-1}]"); continue

            window_norm = x_norm[kk - p:kk]
            region = "train" if kk < train_end else "test"
            print(f"  k={kk} ({region})")
            show_prediction(window_norm, model, xmin, xmax, true_val=x[kk])

        elif len(parts) == p:
            try:
                raw = np.array([float(v) for v in parts])
            except ValueError:
                print("Ошибка: нужны числа"); continue

            window_norm = normalize(raw, xmin, xmax)
            print(f"  своё окно: {raw}")
            show_prediction(window_norm, model, xmin, xmax, true_val=None)

        else:
            print(f"Нужно либо одно число (k), либо {p} чисел (окно)")

if __name__ == "__main__":
    k, x = gen_series()
    xmin, xmax = x[:TRAIN_END].min(), x[:TRAIN_END].max()
    x_norm = normalize(x, xmin, xmax)

    print(f"Train: k=0..{TRAIN_END-1} ({TRAIN_END} точек); Test: k={TRAIN_END}..{K-1} ({K-TRAIN_END} точек)")
    print(f"x_min(train)={xmin:.4f}, x_max(train)={xmax:.4f}")
    print(f"x_norm(test) диапазон: [{x_norm[TRAIN_END:].min():.4f}; {x_norm[TRAIN_END:].max():.4f}]  "
          f"(выход за [0;1] ожидаем, т.к. min/max считаны только по train)")

    plt.figure(figsize=(12, 4))
    plt.plot(k, x)
    plt.axvline(TRAIN_END - 0.5, linestyle="--", color="black", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title("Ряд x(k), вариант 6")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot0_series.png", dpi=110)
    plt.show()

    Xtr, Ytr, ktr = make_windows(x_norm, p_var, p_var, TRAIN_END - 1)
    Xte, Yte, kte = make_windows(x_norm, p_var, TRAIN_END, K - 1)
    print(f"Обучающих примеров: {len(Xtr)}; тестовых примеров: {len(Xte)}")

    # базовый эксперимент
    base_model, base_history = train(Xtr, Ytr, seed=1, p=p_var, N=N_var)
    mae_tr, rmse_tr, smape_tr, *_ = metrics(base_model, Xtr, Ytr, xmin, xmax)
    mae_te, rmse_te, smape_te, preds_te_base, true_te_base = metrics(base_model, Xte, Yte, xmin, xmax)
    print(f"\nБазовый эксперимент p={p_var}, N={N_var}, seed=1: "
          f"эпох={len(base_history)}, Es={base_history[-1]:.8f}")
    print(f"Train: MAE={mae_tr:.4f} RMSE={rmse_tr:.4f} sMAPE={smape_tr:.2f}%")
    print(f"Test:  MAE={mae_te:.4f} RMSE={rmse_te:.4f} sMAPE={smape_te:.2f}%")

    # сюда складываем все обученные модели, чтобы потом честно выбрать лучшую по test MAE
    candidates = [(p_var, N_var, 1, base_model, base_history, mae_te)]

    # исследование p (N фиксировано)
    print("\nИсследование p:")
    p_study_history_example = {}
    for p in [2, 4, 6, 12]:
        Xtr_p, Ytr_p, _ = make_windows(x_norm, p, p, TRAIN_END - 1)
        Xte_p, Yte_p, _ = make_windows(x_norm, p, TRAIN_END, K - 1)
        epochs_l, mae_l, nc = [], [], 0
        for seed in range(1, 6):
            m, h = train(Xtr_p, Ytr_p, seed=seed, p=p, N=N_var)
            mae_p, *_ = metrics(m, Xte_p, Yte_p, xmin, xmax)
            epochs_l.append(len(h)); mae_l.append(mae_p)
            if h[-1] > ERRORLIMIT:
                nc += 1
            if p != 2:   # p=2 не сходится вообще, в кандидаты на "лучшую" не берём
                candidates.append((p, N_var, seed, m, h, mae_p))
            if seed == 1 and p in (4, 12):
                p_study_history_example[f"p={p}, N={N_var}"] = h
            print(f"    p={p}, seed={seed}: эпох={len(h)}, "
                  f"{'сошлось' if h[-1] <= ERRORLIMIT else 'НЕ сошлось'}", flush=True)
        print(f"  p={p:2d}: эпох {np.mean(epochs_l):.0f}±{np.std(epochs_l):.0f} | "
              f"test MAE {np.mean(mae_l):.4f}±{np.std(mae_l):.4f} | не сошлось {nc}/5")

    # исследование N (p фиксировано)
    print("\nИсследование N:")
    for N in [1, 2, 4, 8]:
        epochs_l, mae_tr_l, mae_te_l, nc = [], [], [], 0
        for seed in range(1, 6):
            m, h = train(Xtr, Ytr, seed=seed, p=p_var, N=N)
            mae_trN, *_ = metrics(m, Xtr, Ytr, xmin, xmax)
            mae_teN, *_ = metrics(m, Xte, Yte, xmin, xmax)
            epochs_l.append(len(h)); mae_tr_l.append(mae_trN); mae_te_l.append(mae_teN)
            if h[-1] > ERRORLIMIT:
                nc += 1
            candidates.append((p_var, N, seed, m, h, mae_teN))
            print(f"    N={N}, seed={seed}: эпох={len(h)}, "
                  f"{'сошлось' if h[-1] <= ERRORLIMIT else 'НЕ сошлось'}", flush=True)
        print(f"  N={N}: эпох {np.mean(epochs_l):.0f}±{np.std(epochs_l):.0f} | "
              f"train MAE {np.mean(mae_tr_l):.4f} | test MAE {np.mean(mae_te_l):.4f}±{np.std(mae_te_l):.4f} | "
              f"не сошлось {nc}/5")

    # выбираем лучшую по тестовому MAE конфигурацию из всех прогнанных
    candidates.sort(key=lambda c: c[5])
    best_p, best_N, best_seed, best_model, best_history, best_mae = candidates[0]
    print(f"\nЛучшая конфигурация по тестовому MAE: p={best_p}, N={best_N}, seed={best_seed}, "
          f"test MAE={best_mae:.4f}, эпох={len(best_history)}")

    Xtr_b, Ytr_b, ktr_b = make_windows(x_norm, best_p, best_p, TRAIN_END - 1)
    Xte_b, Yte_b, kte_b = make_windows(x_norm, best_p, TRAIN_END, K - 1)
    mae_trb, rmse_trb, smape_trb, preds_trb, true_trb = metrics(best_model, Xtr_b, Ytr_b, xmin, xmax)
    mae_teb, rmse_teb, smape_teb, preds_teb, true_teb = metrics(best_model, Xte_b, Yte_b, xmin, xmax)
    print(f"Train (лучшая): MAE={mae_trb:.4f} RMSE={rmse_trb:.4f} sMAPE={smape_trb:.2f}%")
    print(f"Test  (лучшая): MAE={mae_teb:.4f} RMSE={rmse_teb:.4f} sMAPE={smape_teb:.2f}%")

    # рекурсивный прогноз на 30 шагов для лучшей конфигурации
    rec_preds_norm = recursive_forecast(best_model, x_norm, best_p, TRAIN_END, 30)
    rec_preds = denormalize(rec_preds_norm, xmin, xmax)
    true_rec = x[TRAIN_END:TRAIN_END + 30]
    rec_err = np.abs(rec_preds - true_rec)
    data_range = x[:TRAIN_END].max() - x[:TRAIN_END].min()
    threshold = 0.05 * data_range
    exceed_idx = np.where(rec_err > threshold)[0]
    first_exceed = exceed_idx[0] + 1 if len(exceed_idx) > 0 else None
    print(f"\nРекурсивный прогноз (p={best_p}, N={best_N}): размах train={data_range:.4f}, "
          f"порог 5%={threshold:.4f}")
    print(f"Первый шаг превышения порога: {first_exceed if first_exceed else 'не превышен за 30 шагов'}")

    # графики
    plt.figure(figsize=(12, 5))
    plt.plot(k, x, label="Исходный ряд", color="gray", alpha=0.6)
    plt.plot(ktr_b, preds_trb, label="Прогноз (train)", color="blue")
    plt.plot(kte_b, preds_teb, label="Прогноз (test)", color="red")
    plt.axvline(TRAIN_END - 0.5, linestyle="--", color="black", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title(f"Ряд и одношаговый прогноз (лучшая конфигурация p={best_p}, N={best_N})")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot1_series_forecast.png", dpi=110)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(base_history, label="Базовая (p=6,N=4)")
    for label, hist in p_study_history_example.items():
        plt.plot(hist, label=label)
    plt.axhline(ERRORLIMIT, linestyle="--", color="black", label=f"Ee={ERRORLIMIT}")
    plt.yscale("log")
    plt.xlabel("Эпоха"); plt.ylabel("Es (лог. шкала)")
    plt.title("Сходимость: базовая + 2 конфигурации из исследования p")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot2_convergence.png", dpi=110)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_teb, preds_teb, alpha=0.7)
    lims = [min(true_teb.min(), preds_teb.min()), max(true_teb.max(), preds_teb.max())]
    plt.plot(lims, lims, "k--", label="y = x")
    plt.xlabel("Эталон"); plt.ylabel("Прогноз")
    plt.title("Эталон vs прогноз (тест, лучшая конфигурация)")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot3_scatter.png", dpi=110)
    plt.show()

    plt.figure(figsize=(12, 5))
    steps = np.arange(1, 31)
    plt.subplot(1, 2, 1)
    plt.plot(k[TRAIN_END:TRAIN_END + 30], true_rec, label="Эталон", marker="o", ms=3)
    plt.plot(k[TRAIN_END:TRAIN_END + 30], rec_preds, label="Рекурсивный прогноз", marker="x", ms=3)
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title(f"Рекурсивный прогноз, 30 шагов (p={best_p}, N={best_N})")
    plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(steps, rec_err, marker="o", ms=3)
    plt.axhline(threshold, linestyle="--", color="red", label="5% размаха")
    plt.xlabel("Номер шага"); plt.ylabel("|ошибка|")
    plt.title("Рост ошибки рекурсивного прогноза")
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig("plot4_recursive.png", dpi=110)
    plt.show()

    # демонстрация режима функционирования (не менее 3 train + 3 test + 1-2 своих окон)
    print(f"\nДемонстрация режима функционирования (лучшая модель p={best_p}, N={best_N}):")

    train_demo_ks = [20, 70, 130]
    test_demo_ks = [150, 170, 195]

    for kk in train_demo_ks + test_demo_ks:
        region = "train" if kk < TRAIN_END else "test"
        print(f"  k={kk} ({region})")
        show_prediction(x_norm[kk - best_p:kk], best_model, xmin, xmax, true_val=x[kk])

    custom_outlier = x[50:50 + best_p].copy()
    custom_outlier[best_p // 2] += 3.0
    print("  своё окно 1 (с выбросом)")
    show_prediction(normalize(custom_outlier, xmin, xmax), best_model, xmin, xmax)

    custom_flat = np.full(best_p, 0.3)
    print("  своё окно 2 (ровное, norm=0.3)")
    show_prediction(custom_flat, best_model, xmin, xmax)

    print("Все графики сохранены (plot0..plot4_*.png) в рабочей директории.\n")

    # можно поиграться с сетью вручную
    functioning_mode(best_model, best_p, x, x_norm, xmin, xmax, TRAIN_END)