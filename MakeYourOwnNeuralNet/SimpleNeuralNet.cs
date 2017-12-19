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
    }
}
