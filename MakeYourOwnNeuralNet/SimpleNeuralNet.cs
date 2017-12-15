using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace MakeYourOwnNeuralNet
{
    class SimpleNeuralNet
    {
        public delegate double ActivationFunctionDelegate(double x);
        private int NumberOfInputNodes;
        private int NumberOfHiddenNodes;
        private int NumberOfOutputNodes;
        private Matrix<double> W_input_hidden;
        private Matrix<double> W_hiddhen_output;
        private Random rndGen;
        
        public ActivationFunctionDelegate ActivationFunction { set; get; }
        public double LearningRate { set; get; }

        public SimpleNeuralNet(int NumberOfInputNodes, int NumberOfHiddenNodes, int NumberOfOutputNodes)
        {
            this.NumberOfInputNodes = NumberOfInputNodes;
            this.NumberOfHiddenNodes = NumberOfHiddenNodes;
            this.NumberOfOutputNodes = NumberOfOutputNodes;
            rndGen = new Random();
            ActivationFunction = Sigmoid;
            LearningRate = 0.3;
        }

        public void Initialize()
        {
            W_input_hidden = DenseMatrix.Build.Dense(NumberOfInputNodes, NumberOfHiddenNodes);
            W_hiddhen_output = DenseMatrix.Build.Dense(NumberOfHiddenNodes, NumberOfOutputNodes);
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

            W_hiddhen_output += LearningRate * ((errors_output * output * (DenseVector.OfArray(new double[] { 1.0, 1.0, 1.0 }) - output)) * OWIH);
            W_input_hidden += LearningRate * ((errors_hidden * OWIH * (DenseVector.OfArray(new double[] { 1.0, 1.0, 1.0 }) - OWIH)) * input);
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
            Console.WriteLine("Initialize the network...");
            var net = new SimpleNeuralNet(3, 3, 3);
            net.Initialize();
            Console.WriteLine(net.Query(DenseVector.OfArray(new double[] { 1.0, 0.5, -1.5 })));
        }
    }
}
