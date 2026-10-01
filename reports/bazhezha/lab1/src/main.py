import numpy as np

# Вариант 1 (Головко: сигмоида F:[0,1] => нормируем классы)
C0, C1 = 0.0, -6.0
ALPHA, EE, NMAX = 0.7, 0.01, 50000  # (3.6): 0<α<1; стоп Es≤Ee
SEED = 0  # разная инициализация может не сойтись (§3.7)


def sigmoid(S):
    S = np.clip(S, -500, 500)
    return 1.0 / (1.0 + np.exp(-S))  # (3.32)


def nrm(x):
    return (x - C0) / (C1 - C0)


def den(y):
    return C0 + y * (C1 - C0)


def cls(y):
    return C0 if abs(y - C0) <= abs(y - C1) else C1


# XOR, вариант 1: (0,0)->0, (0,-6)->-6, (-6,0)->-6, (-6,-6)->0
X = nrm(np.array([[0, 0], [0, -6], [-6, 0], [-6, -6]], dtype=float))
E = nrm(np.array([0, -6, -6, 0], dtype=float))  # эталоны e


def es(net):
    # (3.12): Es = 1/2 Σ_k (y^k - e^k)^2
    return 0.5 * np.sum((np.array([net.fwd(x) for x in X]) - E) ** 2)


def acc(net):
    ok = sum(cls(den(net.fwd(x))) == cls(den(e)) for x, e in zip(X, E))
    return ok / len(E)


def fit(net):
    # алгоритм 3.6, последовательное обучение
    for ep in range(1, NMAX + 1):
        for x, e in zip(X, E):
            net.step(x, e)
        s = es(net)
        if s <= EE:
            return ep, s, True
    return NMAX, es(net), False


class MLP:
    """Персептрон 2-2-1, сигмоида, online backprop (Головко, гл. 3)."""

    def __init__(self):
        r = np.random.default_rng(SEED)
        # §3.6 п.2: веса и пороги случайно в узком диапазоне, напр. [-0.1, 0.1]
        self.Wh = r.uniform(-0.1, 0.1, (2, 2))  # скрытый: 2 входа -> 2 нейрона
        self.Th = r.uniform(-0.1, 0.1, 2)
        self.Wo = r.uniform(-0.1, 0.1, 2)       # выход: 2 скрытых -> 1 нейрон
        self.To = float(r.uniform(-0.1, 0.1))

    def fwd(self, x):
        # (3.33), (3.32): S = Σ ω y − T;  y = 1/(1+e^{-S})
        self.x = x
        self.Sh = x @ self.Wh - self.Th
        self.h = sigmoid(self.Sh)
        self.So = float(self.h @ self.Wo - self.To)
        self.y = float(sigmoid(self.So))
        return self.y

    def step(self, x, e):
        y = self.fwd(x)
        # (3.37): γ_вых = y − e
        go = y - e
        dFo = y * (1.0 - y)  # (3.34): F'(S) = y(1−y)
        # (3.35), (3.36) выходной слой
        self.Wo -= ALPHA * go * dFo * self.h
        self.To += ALPHA * go * dFo
        # (3.38): γ_скр = γ_вых · F'(S_вых) · ω
        gh = go * dFo * self.Wo
        dFh = self.h * (1.0 - self.h)
        # (3.35), (3.36) скрытый слой
        self.Wh -= ALPHA * np.outer(self.x, gh * dFh)
        self.Th += ALPHA * gh * dFh


class SLP:
    """Однослойный персептрон, один сигмоидный нейрон (Головко, гл. 2–3)."""

    def __init__(self):
        r = np.random.default_rng(SEED)
        self.W = r.uniform(-0.1, 0.1, 2)
        self.T = float(r.uniform(-0.1, 0.1))

    def fwd(self, x):
        self.x = x
        self.S = float(x @ self.W - self.T)  # (2.3), (3.33)
        self.y = float(sigmoid(self.S))
        return self.y

    def step(self, x, e):
        y = self.fwd(x)
        g = y - e  # (3.37)
        dF = y * (1.0 - y)
        self.W -= ALPHA * g * dF * self.x  # (3.35)
        self.T += ALPHA * g * dF           # (3.36)


def report(name, net, ep, s, ok):
    print(f"\n{name}")
    print(f"epochs={ep}  Es={s:.6f}  stop={'yes' if ok else 'no'}  acc={acc(net):.0%}")
    for x, e in zip(X, E):
        y = net.fwd(x)
        print(f"  {den(x[0]):.0f} {den(x[1]):.0f} -> {den(y):.4f}  t={den(e):.0f}  {cls(den(y)):.0f}")


def main():
    mlp, slp = MLP(), SLP()
    report("MLP 2-2-1", mlp, *fit(mlp))
    report("SLP 2-1", slp, *fit(slp))
    print("\nXOR линейно неразделима: SLP не сходится, MLP решает задачу.")
    print("A B из [-10,10] (пусто = выход)")
    while True:
        s = input("> ").strip()
        if not s:
            break
        try:
            a, b = map(float, s.split())
        except ValueError:
            print("нужно: A B")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("вне диапазона")
            continue
        y = den(mlp.fwd(np.array([nrm(a), nrm(b)])))
        tag = "c0" if cls(y) == C0 else "c1"
        print(f"y={y:.4f}  class={cls(y):.0f} ({tag})")


if __name__ == "__main__":
    main()
