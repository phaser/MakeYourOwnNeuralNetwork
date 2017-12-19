using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.IO;

namespace MakeYourOwnNeuralNet
{
    class SimpleNeuralNet
    {
        public delegate double ActivationFunctionDelegate(double x);
        private int NumberOfInputNodes;
        private int NumberOfHiddenNodes;
        private int NumberOfOutputNodes;
        private DenseVector one_vector_output;
        private Matrix<double> W_input_hidden;
        private Matrix<double> W_hidden_output;
        private DenseVector one_vector_hidden;
        private Random rndGen;
        
        public ActivationFunctionDelegate ActivationFunction { set; get; }
        public double LearningRate { set; get; }

        public SimpleNeuralNet(int NumberOfInputNodes, int NumberOfHiddenNodes, int NumberOfOutputNodes)
        {
            this.NumberOfInputNodes = NumberOfInputNodes;
            this.NumberOfHiddenNodes = NumberOfHiddenNodes;
            this.NumberOfOutputNodes = NumberOfOutputNodes;
            one_vector_output = DenseVector.OfArray(new double[this.NumberOfOutputNodes]);
            for (int i = 0; i < one_vector_output.Count; i++) { one_vector_output[i] = 1.0; }
            one_vector_hidden = DenseVector.OfArray(new double[this.NumberOfHiddenNodes]);
            for (int i = 0; i < one_vector_hidden.Count; i++) { one_vector_hidden[i] = 1.0; }
            rndGen = new Random();
            ActivationFunction = Sigmoid;
            LearningRate = 0.1;
            Console.WriteLine("input nodes: " + this.NumberOfInputNodes);
            Console.WriteLine("hidden nodes: " + this.NumberOfHiddenNodes);
            Console.WriteLine("output nodes: " + this.NumberOfOutputNodes);
            Console.WriteLine("learning rate: " + this.LearningRate);
        }

        public void Initialize()
        {
            W_input_hidden = DenseMatrix.Build.Dense(NumberOfHiddenNodes, NumberOfInputNodes);
            W_hidden_output = DenseMatrix.Build.Dense(NumberOfOutputNodes, NumberOfHiddenNodes);
            InitWeightsWithRandomValues();
            //InitWeightsWithTestValues();
        }

        private void InitWeightsWithTestValues()
        {
            W_input_hidden[0, 0] = 0.9; W_input_hidden[0, 1] = 0.3; W_input_hidden[0, 2] = 0.4;
            W_input_hidden[1, 0] = 0.2; W_input_hidden[1, 1] = 0.8; W_input_hidden[1, 2] = 0.2;
            W_input_hidden[2, 0] = 0.1; W_input_hidden[2, 1] = 0.5; W_input_hidden[2, 2] = 0.6;

            W_hidden_output[0, 0] = 0.3; W_hidden_output[0, 1] = 0.7; W_hidden_output[0, 2] = 0.5;
            W_hidden_output[1, 0] = 0.6; W_hidden_output[1, 1] = 0.5; W_hidden_output[1, 2] = 0.2;
            W_hidden_output[2, 0] = 0.8; W_hidden_output[2, 1] = 0.1; W_hidden_output[2, 2] = 0.9;
        }

        private void InitWeightsWithRandomValues()
        {
            InitMatrixWithRandom(W_input_hidden);
            InitMatrixWithRandom(W_hidden_output);
        }

        public void Train(Vector<double> input, Vector<double> target)
        {
            var OWIH = W_input_hidden * input.ToColumnMatrix();
            OWIH = OWIH.Map(x => ActivationFunction(x));
            var OWHO = W_hidden_output * OWIH;
            OWHO = OWHO.Map(x => ActivationFunction(x));

            var errors_output = target.ToColumnMatrix() - OWHO;
            var errors_hidden = W_hidden_output.Transpose() * errors_output;
            var oneminus = (one_vector_output.ToColumnMatrix() - OWHO);
            var SHO = errors_output.PointwiseMultiply(OWHO).PointwiseMultiply(oneminus);
            var SHO_error = LearningRate * (SHO * OWIH.Transpose());
            W_hidden_output += SHO_error;

            var SIH = errors_hidden.PointwiseMultiply(OWIH).PointwiseMultiply((one_vector_hidden.ToColumnMatrix() - OWIH));
            var SIH_error = LearningRate * (SIH * input.ToRowMatrix());
            W_input_hidden += SIH_error; 
        }

        public Vector<double> Query(Vector<double> input)
        {
            var OWIH = W_input_hidden * input.ToColumnMatrix();
            OWIH = OWIH.Map(x => ActivationFunction(x));
            var OWHO = W_hidden_output * OWIH;
            OWHO = OWHO.Map(x => ActivationFunction(x));
            return DenseVector.OfArray(OWHO.ToColumnMajorArray());
        }

        private void InitMatrixWithRandom(Matrix<double> mat)
        {
            var cols = mat.AsColumnMajorArray();
            for (int i = 0; i < mat.ColumnCount * mat.RowCount; i++)
            {
                cols[i] = rndGen.NextDouble() - 0.49;
            }
        }

        private double Sigmoid(double x)
        {
            double ex = Math.Pow(Math.E, -x);
            return 1 / (ex + 1);
        }

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
                Console.WriteLine("start training: epoch[" + (k+1) + "]");
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
