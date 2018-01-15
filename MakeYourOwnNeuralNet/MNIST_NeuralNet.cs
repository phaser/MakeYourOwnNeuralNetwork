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
            //var net = MNIST_Train();
            //net.SaveModel("MNIST_model.json");
            //net.SaveModel("../MNIST_model.json");
            var net = new SimpleNeuralNet(784, 200, 10);
            net.Initialize();
            net.LoadModel("../MNIST_model.json");
            Kaggle_MNIST_Test(net);
        }

        private static void MNIST_Test(SimpleNeuralNet net)
        {

            Console.WriteLine("Testing...");
            string fname = "../mnist_test.csv";
            //string fname = "../my_test_data.csv";
            //string fname = "../mnist_train.csv";

            var lines = File.ReadAllLines(fname);
            var train_data = new Vector<double>[lines.Length];
            var target_data = new Vector<double>[lines.Length];
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
                Console.WriteLine(right_answer + "  NI: " + max + " " + (max == right_answer ? "R" : "W"));
            }

            Console.WriteLine("Accuracy: " + ((double)right / (double)train_data.Length));
        }

        private static void Kaggle_MNIST_Test(SimpleNeuralNet net)
        {

            Console.WriteLine("Testing...");
            string fname = "../kaggle_mnist_test.csv";
            //string fname = "../my_test_data.csv";
            //string fname = "../mnist_train.csv";

            var lines = File.ReadAllLines(fname);
            var train_data = new Vector<double>[lines.Length - 1];
            var j = -1;
            foreach (var line in lines)
            {
                if (line.Contains("pixel"))
                    continue;
                if (string.IsNullOrEmpty(line))
                    continue;

                var words = line.Split(',');
                string label = words[0];
                double[] nums = new double[words.Length];
                for (int i = 0; i < words.Length; i++)
                {
                    nums[i] = Convert.ToDouble(words[i]);
                    nums[i] = (nums[i] / 255.0) * 0.99 + 0.01; // normalize the nums to be between 0.01 - 1 (inclusive)
                }
                train_data[++j] = DenseVector.OfArray(nums);
            }
            lines = null;

            int right = 0;
            string fileContents = "ImageId,Label\n";
            for (int i = 0; i < train_data.Length; i++)
            {
                var answer = net.Query(train_data[i]);
                int max = 0;
                for (int k = 1; k < answer.Count; k++)
                {
                    if (answer[k] > answer[max])
                    {
                        max = k;
                    }
                }
                fileContents += (i + 1) + "," + max + "\n";
            }
            File.WriteAllText("../kaggle_mnist_answer.csv", fileContents);
        }

        private static SimpleNeuralNet MNIST_Train(int epochs = 10)
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
            for (int k = 0; k < epochs; k++)
            {
                Console.WriteLine("start training: epoch[" + (k + 1) + "]");
                for (int i = 0; i < train_data.Length; i++)
                {
                    net.Train(train_data[i], target_data[i]);
                }
            }
            return net;
        }
    }
}
