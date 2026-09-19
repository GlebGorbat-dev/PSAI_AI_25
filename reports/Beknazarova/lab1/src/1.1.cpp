#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "tabulate/table.hpp"

using namespace std;
using namespace tabulate;

class Perceptron {
private:
    double w1, w2;
    double T;
    double alpha;
    bool trained;
    double c0, c1;

public:
    Perceptron(double lr, double c0_, double c1_)
        : w1(0), w2(0), T(0), alpha(lr), trained(false), c0(c0_), c1(c1_) {
    }

    // c0 -> 0, c1 -> 1
    double normalize(double y) const {
        if (c1 == c0) return 0.0;
        return (y - c0) / (c1 - c0);
    }

    // 0 -> c0, 1 -> c1
    double denormalize(double y_norm) const {
        return c0 + y_norm * (c1 - c0);
    }

    // Сигмоида
    double sigmoid(double x) const {
        return 1.0 / (1.0 + exp(-x));
    }

    // Производная сигмоиды через выход
    double sigmoid_deriv(double y) const {
        return y * (1.0 - y);
    }

    double weightedSum(double x1, double x2) const {
        return w1 * x1 + w2 * x2 - T;
    }

    // Выход в нормализованном виде (сигмоида)
    double predictNorm(double x1, double x2) const {
        return sigmoid(weightedSum(x1, x2));
    }

    double predict(double x1, double x2) const {
        return denormalize(predictNorm(x1, x2));
    }

    double getW1() const { return w1; }
    double getW2() const { return w2; }
    double getT()  const { return T; }

    int epoch(const vector<double>& X1, const vector<double>& X2,
        const vector<double>& e, int epochNum) {
        int errors = 0;
        double Es = 0.0;

        Table table;
        table.add_row({ "N", "x1", "x2", "S", "y_norm", "e_norm",
                        "y_real", "e_real", "w1", "w2", "T" });

        for (size_t i = 0; i < X1.size(); ++i) {
            double x1_val = X1[i];
            double x2_val = X2[i];
            double target_real = e[i];
            double target_norm = normalize(target_real);

            double S = weightedSum(x1_val, x2_val);
            double output_norm = sigmoid(S);

            double err = target_norm - output_norm;
            Es += 0.5 * err * err;

            // Дельта-правило (градиентный спуск по сигмоиде)
            double delta = err * sigmoid_deriv(output_norm);
            w1 += alpha * delta * x1_val;
            w2 += alpha * delta * x2_val;
            T -= alpha * delta;

            if (fabs(err) > 0.2) errors++;

            double output_real = denormalize(output_norm);

            auto fmt = [](double val) {
                ostringstream oss;
                oss << fixed << setprecision(4) << val;
                return oss.str();
                };

            table.add_row({
                to_string(i + 1),
                fmt(x1_val), fmt(x2_val),
                fmt(S),
                fmt(output_norm), fmt(target_norm),
                fmt(output_real), fmt(target_real),
                fmt(w1), fmt(w2), fmt(T)
                });
        }

        cout << "\nЭпоха " << epochNum << "  (Es = " << Es << ")\n";
        cout << table << "\n";
        setlocale(LC_ALL, "ru");
        return errors;
    }

    void train(const vector<double>& X1, const vector<double>& X2,
        const vector<double>& Y, int maxEpochs) {
        for (int epochNum = 1; epochNum <= maxEpochs; ++epochNum) {
            int err = epoch(X1, X2, Y, epochNum);
            if (err == 0) {
                trained = true;
                cout << "\nОбучение успешно завершено на эпохе " << epochNum << "\n";
                return;
            }
        }
        cout << "\nДостигнут лимит эпох. Выборка, возможно, не линейно разделима.\n";
    }

    bool isTrained() const { return trained; }

    void printWeights() const {
        cout << "w1 = " << w1 << ", w2 = " << w2 << ", T = " << T << endl;
    }
};

int main() {
    setlocale(LC_ALL, "ru");

    double alpha;
    int n, maxEpochs;

    cout << "Введите скорость обучения: ";
    cin >> alpha;
    cout << "Введите макс. число эпох: ";
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

    double c0 = allVals[0];   // меньший
    double c1 = allVals[1];   // больший

    cout << "\nАвтоматически определены классы:\n";
    cout << "  c0 = " << c0 << "\n";
    cout << "  c1 = " << c1 << "\n";

    Perceptron p(alpha, c0, c1);

    cout << "\nНачальные веса: ";
    p.printWeights();

    p.train(x1, x2, e, maxEpochs);

    if (p.isTrained()) {
        cout << "\nУравнение разделяющей прямой:\n";
        cout << "   " << p.getW1() << " * x1 + " << p.getW2()
            << " * x2 - " << p.getT() << " = 0\n";
    }
    else {
        cout << "\nСеть не обучена — проверка не имеет смысла.\n";
    }

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

            double y_norm = p.predictNorm(a, b);
            double y_real = p.predict(a, b);

            double d0 = fabs(y_real - c0);
            double d1 = fabs(y_real - c1);
            string cls = (d0 <= d1) ? "c0" : "c1";

            cout << "   y_norm = " << fixed << setprecision(4) << y_norm
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