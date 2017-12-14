using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace MakeYourOwnNeuralNet
{
    class SimpleNeuralNet
    {
        private int NumberOfInputNodes;
        private int NumberOfHiddenNodes;
        private int NumberOfOutputNodes;
        private Matrix<double> W_input_hidden;
        private Matrix<double> W_hiddhen_output;
        private Random rndGen;

        public SimpleNeuralNet(int NumberOfInputNodes, int NumberOfHiddenNodes, int NumberOfOutputNodes)
        {
            this.NumberOfInputNodes = NumberOfInputNodes;
            this.NumberOfHiddenNodes = NumberOfHiddenNodes;
            this.NumberOfOutputNodes = NumberOfOutputNodes;
            rndGen = new Random();
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

        private void InitMatrixWithRandom(Matrix<double> mat)
        {
            var cols = mat.AsColumnMajorArray();
            for (int i = 0; i < mat.ColumnCount * mat.RowCount; i++)
            {
               cols[i] = rndGen.NextDouble();
            }
        }

        public void Train()
        {

        }

        static void Main(string[] args)
        {
            Console.WriteLine("Initialize the network...");
            var net = new SimpleNeuralNet(3, 3, 3);
            net.Initialize();
        }
    }
}
