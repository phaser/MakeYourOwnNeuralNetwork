using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.IO;

namespace MakeYourOwnNeuralNet
{
    class MNIST_NeuralNet
    {
        static void Main(string[] args)
        {
            MNIST_Train();
        }

        private static void MNIST_Train()
        {
            var lines = File.ReadAllLines("../mnist_train.csv");
            Vector<double>[] train_data = new Vector<double>[lines.Length];
            Vector<double>[] target_data = new Vector<double>[lines.Length];
            var j = -1;
            foreach (var line in lines)
            {
                var words = line.Split(',');
                string label = words[0];
                double[] nums = new double[words.Length - 1];
                for (int i = 1; i < words.Length; i++)
                {
                    nums[i - 1] = Convert.ToDouble(words[i]);
                    nums[i - 1] = (nums[i - 1] / 255.0) * 0.99 + 0.01; // normalize the nums to be between 0.01 - 1 (inclusive)
                }
                train_data[++j] = DenseVector.OfArray(nums);
                var target = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    target[i] = 0.01;
                }
                target[Convert.ToInt32(label)] = 0.99;
                target_data[j] = DenseVector.OfArray(target);
            }

            Console.WriteLine("Initialize the network...");
            var net = new SimpleNeuralNet(train_data[0].Count, 200, 10);
            net.Initialize();
            for (int k = 0; k < 5; k++)
            {
                Console.WriteLine("start training: epoch[" + (k + 1) + "]");
                for (int i = 0; i < train_data.Length; i++)
                {
                    net.Train(train_data[i], target_data[i]);
                }
            }

            Console.WriteLine("Testing...");
            lines = File.ReadAllLines("../mnist_test.csv");
            train_data = new Vector<double>[lines.Length];
            target_data = new Vector<double>[lines.Length];
            j = -1;
            foreach (var line in lines)
            {
                var words = line.Split(',');
                string label = words[0];
                double[] nums = new double[words.Length - 1];
                for (int i = 1; i < words.Length; i++)
                {
                    nums[i - 1] = Convert.ToDouble(words[i]);
                    nums[i - 1] = (nums[i - 1] / 255.0) * 0.99 + 0.01; // normalize the nums to be between 0.01 - 1 (inclusive)
                }
                train_data[++j] = DenseVector.OfArray(nums);
                var target = new double[10];
                for (int i = 0; i < 10; i++)
                {
                    target[i] = 0.01;
                }
                target[Convert.ToInt32(label)] = 0.99;
                target_data[j] = DenseVector.OfArray(target);
            }
            lines = null;

            int right = 0;
            for (int i = 0; i < train_data.Length; i++)
            {
                Console.WriteLine("Test #" + (i + 1));
                int right_answer = 0;
                for (int k = 0; k < target_data[i].Count; k++)
                {
                    if (target_data[i][k] > 0.98)
                    {
                        right_answer = k;
                        break;
                    }
                }

                var answer = net.Query(train_data[i]);
                int max = 0;
                for (int k = 1; k < answer.Count; k++)
                {
                    if (answer[k] > answer[max])
                    {
                        max = k;
                    }
                }

                right += (max == right_answer ? 1 : 0);
                Console.WriteLine("Result: " + max + " " + right_answer + " " + (max == right_answer ? "RIGHT" : "WRONG"));
            }

            Console.WriteLine("Accuracy: " + ((double)right / (double)train_data.Length));
        }
    }
}
