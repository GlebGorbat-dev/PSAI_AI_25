using System;

namespace XOR_NeuralNetwork
{
    public class NeuralNetwork
    {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;

        private double[,] weightsInputHidden;
        private double[] biasHidden;
        private double[] hiddenOutputs;

        private double[] weightsHiddenOutput;
        private double biasOutput;

        private Random rnd = new Random(42);

        public NeuralNetwork(int input, int hidden, int output)
        {
            inputSize = input;
            hiddenSize = hidden;
            outputSize = output;

            if (hiddenSize > 0)
            {
                weightsInputHidden = new double[inputSize, hiddenSize];
                biasHidden = new double[hiddenSize];
                hiddenOutputs = new double[hiddenSize];
                InitializeArray(weightsInputHidden);
                InitializeArray(biasHidden);
            }

            weightsHiddenOutput = new double[hiddenSize > 0 ? hiddenSize : inputSize];
            biasOutput = 0;
            InitializeArray(weightsHiddenOutput);
        }

        private void InitializeArray(double[,] array)
        {
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    array[i, j] = rnd.NextDouble();
        }

        private void InitializeArray(double[] array)
        {
            for (int i = 0; i < array.Length; i++)
                array[i] = rnd.NextDouble();
        }

        private double Sigmoid(double x)
        {
            if (x < -45) return 0;
            if (x > 45) return 1;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double output)
        {
            return output * (1.0 - output);
        }

        public double Forward(double[] inputs)
        {
            if (hiddenSize > 0)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    double sum = biasHidden[j];
                    for (int i = 0; i < inputSize; i++)
                        sum += inputs[i] * weightsInputHidden[i, j];
                    hiddenOutputs[j] = Sigmoid(sum);
                }

                double finalSum = biasOutput;
                for (int j = 0; j < hiddenSize; j++)
                    finalSum += hiddenOutputs[j] * weightsHiddenOutput[j];
                return Sigmoid(finalSum);
            }
            else
            {
                double sum = biasOutput;
                for (int i = 0; i < inputSize; i++)
                    sum += inputs[i] * weightsHiddenOutput[i];
                return Sigmoid(sum);
            }
        }

        public double Train(double[] inputs, double target, double learningRate)
        {
            double output = Forward(inputs);
            double error = target - output;
            double outputDelta = error * SigmoidDerivative(output);

            if (hiddenSize > 0)
            {
                for (int j = 0; j < hiddenSize; j++)
                    weightsHiddenOutput[j] += learningRate * outputDelta * hiddenOutputs[j];
                biasOutput += learningRate * outputDelta;

                double[] hiddenDeltas = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    double errorHidden = outputDelta * weightsHiddenOutput[j];
                    hiddenDeltas[j] = errorHidden * SigmoidDerivative(hiddenOutputs[j]);
                }

                for (int j = 0; j < hiddenSize; j++)
                {
                    for (int i = 0; i < inputSize; i++)
                        weightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
                    biasHidden[j] += learningRate * hiddenDeltas[j];
                }
            }
            else
            {
                for (int i = 0; i < inputSize; i++)
                    weightsHiddenOutput[i] += learningRate * outputDelta * inputs[i];
                biasOutput += learningRate * outputDelta;
            }

            return error * error;
        }
    }

    class Program
    {
        static double Normalize(double x) => (x + 10.0) / 20.0;
        static double Denormalize(double y) => y * 20.0 - 10.0;

        static void Main(string[] args)
        {
            

            double[][] rawInputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, -6 },
                new double[] { -6, 0 },
                new double[] { -6, -6 }
            };

            double[] rawTargets = new double[] { 0, -6, -6, 0 };

            double[][] inputs = new double[rawInputs.Length][];
            double[] targets = new double[rawTargets.Length];

            for (int i = 0; i < rawInputs.Length; i++)
            {
                inputs[i] = new double[] { Normalize(rawInputs[i][0]), Normalize(rawInputs[i][1]) };
                targets[i] = Normalize(rawTargets[i]);
            }

            Console.WriteLine("Нормализованные данные:");
            for (int i = 0; i < inputs.Length; i++)
                Console.WriteLine($"  [{inputs[i][0]:F2}, {inputs[i][1]:F2}] -> {targets[i]:F2}");

            double learningRate = 2.0;
            double targetError = 0.01;

            Console.WriteLine("\n Обучение MLP (2-2-1) ");
            NeuralNetwork mlp = new NeuralNetwork(2, 2, 1);
            TrainUntilConvergence(mlp, inputs, targets, learningRate, targetError, "MLP", 200000);

            Console.WriteLine("\n Обучение Однослойного перцептрона (2-1) ");
            NeuralNetwork singleLayer = new NeuralNetwork(2, 0, 1);
            TrainUntilConvergence(singleLayer, inputs, targets, learningRate, targetError, "Single Layer", 20000);

            Console.WriteLine("\n Итоговые результаты ");
            Console.WriteLine("\nMLP (2-2-1):");
            PrintResults(mlp, inputs, targets, rawInputs, rawTargets);

            Console.WriteLine("\nSingle Layer (2-1):");
            PrintResults(singleLayer, inputs, targets, rawInputs, rawTargets);

            Console.WriteLine("\n Режим тестирования ");
            Console.WriteLine("Введите два числа (A и B) в диапазоне [-10; 10] через пробел :");

            while (true)
            {
                string input = Console.ReadLine();
                if (input == null || input.Trim().ToLower() == "q") break;

                string[] parts = input.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 2) continue;

                if (double.TryParse(parts[0], out double a) && double.TryParse(parts[1], out double b))
                {
                    double[] testInput = new double[] { Normalize(a), Normalize(b) };
                    double rawOut = mlp.Forward(testInput);
                    double denormOut = Denormalize(rawOut);

                    Console.WriteLine($"MLP выход: {denormOut:F4}");

                    double dist0 = Math.Abs(denormOut - 0);
                    double distM6 = Math.Abs(denormOut - (-6));
                    string cls = dist0 < distM6 ? "0" : "-6";
                    Console.WriteLine($"Ближе к классу: {cls}");
                }
            }
        }

        static void TrainUntilConvergence(NeuralNetwork net, double[][] inputs, double[] targets,
            double lr, double targetError, string name, int safetyLimit)
        {
            int epoch = 0;
            double totalError = double.MaxValue;

            while (totalError > targetError && epoch < safetyLimit)
            {
                totalError = 0;
                for (int i = 0; i < inputs.Length; i++)
                    totalError += net.Train(inputs[i], targets[i], lr);

                if (epoch % 1000 == 0)
                    Console.WriteLine($"{name} - Эпоха: {epoch}, Ошибка: {totalError:F6}");

                epoch++;
            }

            if (totalError <= targetError)
                Console.WriteLine($"{name} Сходимость на эпохе {epoch}. Ошибка: {totalError:F6}");
            else
                Console.WriteLine($"{name} Достигнут лимит {safetyLimit} эпох. Ошибка: {totalError:F6}");
        }

        static void PrintResults(NeuralNetwork net, double[][] inputs, double[] targets,
            double[][] rawInputs, double[] rawTargets)
        {
            Console.WriteLine("A\tB\tОжидалось\tПолучено\tКласс");
            for (int i = 0; i < inputs.Length; i++)
            {
                double output = net.Forward(inputs[i]);
                double denormOutput = Denormalize(output);

                double dist0 = Math.Abs(denormOutput - 0);
                double distM6 = Math.Abs(denormOutput - (-6));
                string predictedClass = dist0 < distM6 ? "0" : "-6";

                Console.WriteLine($"{rawInputs[i][0]}\t{rawInputs[i][1]}\t{rawTargets[i]}\t\t{denormOutput:F4}\t\t{predictedClass}");
            }
        }
    }
}