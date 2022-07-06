using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public class Neuron
    {
        private double bias;
        private double output;
        private List<double> weights = new List<double>();
        private List<double> inputs = new List<double>();
        private double learnRate;
        public double errorGradient;
        private Random rnd;

        public Neuron(int numInputs, double LearnRate)
        {
            rnd = new Random();

            learnRate = LearnRate;

            for (int i = 0; i < numInputs; i++)
            {
                weights.Add(rnd.NextDouble());
            }

            bias = rnd.NextDouble();
        }

        private static float Sigmoid(double value)
        {
            return (float)(1.0 / (1.0 + Math.Pow(Math.E, -value)));
        }

        public void SetInputs(List<double> input)
        {
            if (input.Count != weights.Count)
            {
                //Console.WriteLine("Wrong number of inputs, need " + weights.Count + " inputs");
                return;
            }
            inputs = input;
        }

        public void AddInput(double input)
        {
            inputs.Add(input);
        }

        public double CalcOutput()
        {
            double total = 0;

            if (inputs.Count != weights.Count)
            {
                //Console.WriteLine("Incorrect number of inputs, should be " + weights.Count + " inputs, there is " + inputs.Count + " inputs");
                return 0;
            }

            for (int i = 0; i < inputs.Count; i++)
            {
                total += inputs[i] * weights[i];
            }

            output = Sigmoid(total - bias);

            return output;
        }

        public void ReWeightOutput(double desiredOutput, double output)
        {
            double error = desiredOutput - output;
            errorGradient = output * (1 - output) * error;

            for (int i = 0; i < inputs.Count; i++)
            {
                weights[i] += learnRate * inputs[i] * error;
            }

            bias += learnRate * -1 * errorGradient;
        }

        public void ReWeightHidden(Layer prevLayer, int weightNum)
        {
            errorGradient = output * (1 - output);
            double errorGradSum = 0;

            for (int i = 0; i < prevLayer.neurons.Count; i++)
            {
                errorGradSum += prevLayer.neurons[i].errorGradient * prevLayer.neurons[i].weights[weightNum];
            }

            errorGradient *= errorGradSum;

            for (int i = 0; i < inputs.Count; i++)
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

            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] = Convert.ToDouble(arr[i]);
            }

            bias = Convert.ToDouble(arr[weights.Count]);
        }

        public void Mutate()
        {
            for(int i = 0; i < weights.Count; i++)
            {
                weights[i] += weights[i] * 0.001f * rnd.Next(-100, 100);
            }
        }
    }
}
