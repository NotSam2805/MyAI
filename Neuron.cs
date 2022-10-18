using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public class Neuron
    {
        private double bias;
        private double output;
        private double[] weights;
        private double[] inputs;
        private double learnRate;
        public double errorGradient;
        private Random rnd;

        public Neuron(int numInputs, double LearnRate)
        {
            rnd = new Random();

            learnRate = LearnRate;

            inputs = new double[numInputs];
            weights = new double[numInputs];

            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.Next(-100,100)/(double)100;
            }

            bias = rnd.NextDouble();
        }

        private static float Sigmoid(double value)
        {
            return (float)(1.0 / (1.0 + Math.Pow(Math.E, -((value + 0) * 1))));
        }

        private static float TanH(double value)
        {
            return (float)((2 / (1 + Math.Exp(-2 * value))) - 1);
        }

        public void SetInputs(double[] input)
        {
            if (input.Length != weights.Length)
            {
                //Console.WriteLine("Wrong number of inputs, need " + weights.Count + " inputs");
                return;
            }
            inputs = input;
        }

        public double CalcOutput()
        {
            double total = 0;

            if (inputs.Length != weights.Length)
            {
                //Console.WriteLine("Incorrect number of inputs, should be " + weights.Length + " inputs, there is " + inputs.Count + " inputs");
                return 0;
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                total += inputs[i] * weights[i];
            }

            output = Sigmoid(total - bias);

            return output;
        }

        public double CalcOutputHidden()
        {
            double total = 0;

            if (inputs.Length != weights.Length)
            {
                //Console.WriteLine("Incorrect number of inputs, should be " + weights.Count + " inputs, there is " + inputs.Count + " inputs");
                return 0;
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                total += inputs[i] * weights[i];
            }

            output = TanH(total - bias);

            return output;
        }

        public void ReWeightOutput(double desiredOutput)
        {
            double error = desiredOutput - output;
            errorGradient = output * (1 - output) * error;

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * error;
            }

            bias += learnRate * -1 * errorGradient;
        }

        public void ReWeightOutput(double desiredOutput, double output)
        {
            double error = desiredOutput - output;
            errorGradient = output * (1 - output) * error;

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * error;
            }

            bias += learnRate * -1 * errorGradient;
        }

        public void ReWeightHidden(Layer prevLayer, int weightNum)
        {
            errorGradient = output * (1 - output);
            double errorGradSum = 0;

            for (int i = 0; i < prevLayer.neurons.Length; i++)
            {
                errorGradSum += prevLayer.neurons[i].errorGradient * prevLayer.neurons[i].weights[weightNum];
            }

            errorGradient *= errorGradSum;

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * errorGradient;
            }

            bias += learnRate * -1 * errorGradient;
        }
        
        public string GetWeightsString()
        {
            string text = "";

            foreach (double weight in weights)
            {
                text += weight.ToString("0.00000") + ",";
            }

            text += bias.ToString("0.00000");

            return text;
        }

        public void SetWeights(string weightsString)
        {
            string[] arr = weightsString.Split(',');

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Convert.ToDouble(arr[i]);
            }

            bias = Convert.ToDouble(arr[weights.Length]);
        }

        public void Mutate()
        {
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] += weights[i] * 0.001f * rnd.Next(-100, 100);
            }
            bias += bias * 0.001f * rnd.Next(-100, 100);
        }
    }
}
