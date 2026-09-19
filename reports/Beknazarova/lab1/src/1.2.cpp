#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "tabulate/table.hpp"

using namespace std;
using namespace tabulate;

static mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

// Сигмоидная функция активации
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Производная сигмоиды, выраженная через выход
double sigmoid_deriv(double y) { return y * (1.0 - y); }

// Случайная инициализация весов малыми значениями [-0.5, 0.5]
double rand_small() {
    uniform_real_distribution<double> dist(-0.5, 0.5);
    return dist(rng);
}

class MLP {
private:
    double W1[2][2];   // скрытый слой: 2 нейрона × 2 входа
    double T1[2];      // пороги скрытого слоя
    double W2[2];      // выходной слой
    double T2;         // порог выходного слоя
    double lr;
    double c0, c1;
    double x_min, x_max;
    bool trained;

public:
    int epochs_trained;
    double final_error;

    MLP(double lr_, double c0_, double c1_)
        : lr(lr_), c0(c0_), c1(c1_), trained(false),
        epochs_trained(0), final_error(0) {
        x_min = min(c0, c1);
        x_max = max(c0, c1);

        // Инициализация весов и порогов случайными малыми значениями
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j)
                W1[i][j] = rand_small();
            T1[i] = rand_small();
            W2[i] = rand_small();
        }
        T2 = rand_small();
    }

    // Нормализация входа: c0/c1 -> [-1, 1]
    double normalizeInput(double x) const {
        if (x_max == x_min) return 0.0;
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0;
    }

    // Нормализация выхода: c0/c1 -> [0, 1]
    double normalizeOutput(double y) const {
        if (c1 == c0) return 0.0;
        return (y - c0) / (c1 - c0);
    }

    // Денормализация: [0, 1] -> c0/c1
    double denormalizeOutput(double y_norm) const {
        return c0 + y_norm * (c1 - c0);
    }

    // Прямой проход
    void forward(double x1n, double x2n, double& h1, double& h2, double& y) {
        double s1 = W1[0][0] * x1n + W1[0][1] * x2n - T1[0];
        double s2 = W1[1][0] * x1n + W1[1][1] * x2n - T1[1];
        h1 = sigmoid(s1);
        h2 = sigmoid(s2);

        double so = W2[0] * h1 + W2[1] * h2 - T2;
        y = sigmoid(so);
    }

    double predictNorm(double x1n, double x2n) {
        double h1, h2, y;
        forward(x1n, x2n, h1, h2, y);
        return y;
    }

    double predict(double x1, double x2) {
        double x1n = normalizeInput(x1);
        double x2n = normalizeInput(x2);
        return denormalizeOutput(predictNorm(x1n, x2n));
    }

    int epoch(const vector<double>& X1, const vector<double>& X2,
        const vector<double>& e, int epochNum, bool verbose) {
        int errors = 0;
        double Es = 0.0;

        Table table;
        table.add_row({ "N", "x1n", "x2n", "h1", "h2",
                        "y_norm", "e_norm", "y_real", "e_real" });

        for (size_t i = 0; i < X1.size(); ++i) {
            double x1n = normalizeInput(X1[i]);
            double x2n = normalizeInput(X2[i]);
            double target_norm = normalizeOutput(e[i]);

            double h1, h2, y;
            forward(x1n, x2n, h1, h2, y);

            double err = target_norm - y;
            Es += 0.5 * err * err;

            // Ошибка выходного нейрона
            double delta_out = err * sigmoid_deriv(y);

            // Ошибки нейронов скрытого слоя
            double delta_h1 = delta_out * W2[0] * sigmoid_deriv(h1);
            double delta_h2 = delta_out * W2[1] * sigmoid_deriv(h2);

            // Онлайн-обновление
            W2[0] += lr * delta_out * h1;
            W2[1] += lr * delta_out * h2;
            T2 -= lr * delta_out;

            W1[0][0] += lr * delta_h1 * x1n;
            W1[0][1] += lr * delta_h1 * x2n;
            T1[0] -= lr * delta_h1;

            W1[1][0] += lr * delta_h2 * x1n;
            W1[1][1] += lr * delta_h2 * x2n;
            T1[1] -= lr * delta_h2;

            if (fabs(y - target_norm) > 0.2) errors++;

            double y_real = denormalizeOutput(y);

            auto fmt = [](double val) {
                ostringstream oss;
                oss << fixed << setprecision(4) << val;
                return oss.str();
                };

            table.add_row({
                to_string(i + 1),
                fmt(x1n), fmt(x2n),
                fmt(h1), fmt(h2),
                fmt(y), fmt(target_norm),
                fmt(y_real), fmt(e[i])
                });
        }

        if (verbose) {
            cout << "\nЭпоха " << epochNum << "  (Es = " << Es << ")\n";
            cout << table << "\n";
            setlocale(LC_ALL, "ru");
        }
        else {
            cout << "Эпоха " << epochNum << "  (Es = " << Es << ")\n";
        }

        final_error = Es;
        epochs_trained = epochNum;
        return errors;
    }

    // Обучение: таблица для первых 10 эпох и каждой 100-й
    void train(const vector<double>& X1, const vector<double>& X2,
        const vector<double>& Y, int maxEpochs, double Ee) {
        for (int epochNum = 1; epochNum <= maxEpochs; ++epochNum) {
            bool verbose = (epochNum <= 10) || (epochNum % 100 == 0);
            int err = epoch(X1, X2, Y, epochNum, verbose);
            if (final_error <= Ee) {
                trained = true;
                cout << "\nОбучение успешно завершено на эпохе " << epochNum
                    << " (Es = " << final_error << " <= " << Ee << ")\n";
                return;
            }
        }
        cout << "\nДостигнут лимит эпох. Es = " << final_error
            << " > Ee = " << Ee << "\n";
    }

    bool isTrained() const { return trained; }

    void printWeights() const {
        cout << "Скрытый слой:\n";
        cout << "  Нейрон 1: W1[0][0]=" << W1[0][0]
            << ", W1[0][1]=" << W1[0][1] << ", T1[0]=" << T1[0] << "\n";
        cout << "  Нейрон 2: W1[1][0]=" << W1[1][0]
            << ", W1[1][1]=" << W1[1][1] << ", T1[1]=" << T1[1] << "\n";
        cout << "Выходной слой:\n";
        cout << "  W2[0]=" << W2[0] << ", W2[1]=" << W2[1]
            << ", T2=" << T2 << "\n";
    }
};

int main() {
    setlocale(LC_ALL, "ru");

    double alpha, Ee;
    int n, maxEpochs;

    cout << "Введите скорость обучения alpha (рекомендую 0.5): ";
    cin >> alpha;
    cout << "Введите порог остановки Ee (рекомендую 0.01): ";
    cin >> Ee;
    cout << "Введите макс. число эпох (рекомендую 50000): ";
    cin >> maxEpochs;
    cout << "Введите количество примеров: ";
    cin >> n;

    vector<double> x1(n), x2(n), e(n);

    cout << "Введите x1 (через пробел): ";
    for (int i = 0; i < n; ++i) cin >> x1[i];

    cout << "Введите x2 (через пробел): ";
    for (int i = 0; i < n; ++i) cin >> x2[i];

    cout << "Введите эталонные значения (через пробел): ";
    for (int i = 0; i < n; ++i) cin >> e[i];

    // Автоматически определяем c0 и c1
    vector<double> allVals;
    for (double v : x1) allVals.push_back(v);
    for (double v : x2) allVals.push_back(v);
    for (double v : e)  allVals.push_back(v);

    sort(allVals.begin(), allVals.end());
    allVals.erase(unique(allVals.begin(), allVals.end()), allVals.end());

    if (allVals.size() < 2) {
        cout << "Ошибка: в данных меньше двух различных классов!\n";
        return 1;
    }

    double c0 = allVals[0];
    double c1 = allVals[1];

    cout << "\nАвтоматически определены классы:\n";
    cout << "  c0 = " << c0 << "\n";
    cout << "  c1 = " << c1 << "\n";

    MLP mlp(alpha, c0, c1);

    cout << "\nНачальные веса и пороги:\n";
    mlp.printWeights();

    mlp.train(x1, x2, e, maxEpochs, Ee);

    cout << "\n--- Итоги обучения ---\n";
    cout << "Эпох: " << mlp.epochs_trained << "\n";
    cout << "Итоговая суммарная ошибка Es: " << mlp.final_error << "\n";

    cout << "\nПроверка на обучающей выборке:\n";
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        double y_real = mlp.predict(x1[i], x2[i]);
        double target_norm = mlp.normalizeOutput(e[i]);
        double y_norm = mlp.normalizeOutput(y_real);

        bool ok = fabs(y_norm - target_norm) < 0.2;
        if (ok) correct++;

        cout << "   (" << x1[i] << ", " << x2[i] << ") -> "
            << "y_norm=" << fixed << setprecision(4) << y_norm
            << ", y_real=" << setprecision(4) << y_real
            << "  [эталон: " << e[i] << "]  "
            << (ok ? "верно" : "не верно") << endl;
    }
    cout << "Accuracy: " << correct << "/" << n
        << " = " << (100.0 * correct / n) << "%\n";

    // Режим функционирования
    cout << "\nРежим функционирования\n";
    cout << "Введите пары (A, B) из [-10; 10]. Для выхода введите 'q'.\n";

    while (true) {
        cout << "\nA B: ";
        string s;
        cin >> s;
        if (s == "q") break;

        try {
            double a = stod(s);
            double b;
            cin >> b;

            double y_real = mlp.predict(a, b);
            double y_norm_klass = mlp.normalizeOutput(y_real);

            double d0 = fabs(y_norm_klass - 0.0);
            double d1 = fabs(y_norm_klass - 1.0);
            string cls = (d0 <= d1) ? "c0" : "c1";

            cout << "   y_norm = " << fixed << setprecision(4) << y_norm_klass
                << ", y_real = " << setprecision(4) << y_real
                << " -> ближайший класс: " << cls << "\n";
        }
        catch (...) {
            cout << "   Ошибка ввода!\n";
            cin.clear();
            cin.ignore(10000, '\n');
        }
    }

    return 0;
}