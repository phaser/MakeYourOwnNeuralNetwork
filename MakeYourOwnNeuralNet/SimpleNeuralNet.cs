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
        private Matrix<double> W_hiddhen_output;
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
            LearningRate = 0.3;
            Console.WriteLine("input nodes: " + this.NumberOfInputNodes);
            Console.WriteLine("hidden nodes: " + this.NumberOfHiddenNodes);
            Console.WriteLine("output nodes: " + this.NumberOfOutputNodes);
            Console.WriteLine("learning rate: " + this.LearningRate);
        }

        public void Initialize()
        {
            W_input_hidden = DenseMatrix.Build.Dense(NumberOfHiddenNodes, NumberOfInputNodes);
            W_hiddhen_output = DenseMatrix.Build.Dense(NumberOfOutputNodes, NumberOfHiddenNodes);
            InitWeightsWithRandomValues();
        }

        private void InitWeightsWithRandomValues()
        {
            InitMatrixWithRandom(W_input_hidden);
            InitMatrixWithRandom(W_hiddhen_output);
        }

        public void Train(Vector<double> input, Vector<double> target)
        {
            var OWIH = W_input_hidden * input;
            OWIH = OWIH.Map(x => ActivationFunction(x));
            var OWHO = W_hiddhen_output * OWIH;
            OWHO = OWHO.Map(x => ActivationFunction(x));

            var output = OWHO;
            var errors_output = target - output;
            var errors_hidden = W_hiddhen_output.Transpose() * errors_output;
            var test = (errors_output * output * (one_vector_output - output));
            W_hiddhen_output += LearningRate * test.ToColumnMatrix() * OWIH.ToRowMatrix();
            W_input_hidden += LearningRate * ((errors_hidden * OWIH * (one_vector_hidden - OWIH)).ToColumnMatrix() * input.ToRowMatrix());
        }

        public Vector<double> Query(Vector<double> input)
        {
            var OWIH = W_input_hidden * input;
            OWIH = OWIH.Map(x => ActivationFunction(x));
            var OWHO = W_hiddhen_output * OWIH;
            OWHO = OWHO.Map(x => ActivationFunction(x));
            return OWHO;
        }

        private void InitMatrixWithRandom(Matrix<double> mat)
        {
            var cols = mat.AsColumnMajorArray();
            for (int i = 0; i < mat.ColumnCount * mat.RowCount; i++)
            {
                cols[i] = rndGen.NextDouble() - 0.5;
            }
        }

        private double Sigmoid(double x)
        {
            double ex = Math.Pow(Math.E, x);
            return ex / (ex + 1);
        }

        static void Main(string[] args)
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
                target[Convert.ToInt32(label)] = 0.99;
                target_data[j] = DenseVector.OfArray(target);
            }

            Console.WriteLine("Initialize the network...");
            var net = new SimpleNeuralNet(train_data[0].Count, 280, 10);
            net.Initialize();
            Console.WriteLine("start training...");
            for (int i = 0; i < train_data.Length; i++)
            {
                Console.WriteLine("#" + i);
                net.Train(train_data[i], target_data[i]);
            }
            // train_data = null; target_data = null; // we don't need this data from now on

            Console.WriteLine(net.Query(train_data[0]));
            Console.WriteLine(target_data[0]);
        }
    }
}
