using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public enum ActivationFunction
    {
        sigmoid,
        tanH
    }

    public class Neuron
    {
        //Self explanaory
        private double bias;
        private double output;
        private double[] weights;
        private double[] inputs;
        
        //Used for something that doesn't work
        private double learnRate;
        public long errorGradient;
        
        //Needed some random numbers
        private Random rnd;

        public Neuron(int numInputs, double LearnRate)
        {
            rnd = new Random();

            learnRate = LearnRate;

            inputs = new double[numInputs];
            weights = new double[numInputs];

            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.Next(-100,100)/(double)100;//Each weight is random between -1 and 1
            }

            bias = rnd.NextDouble();//Random bias
            
            //Might be useful to be able to set the weights and biases here, or even the random seed
        }

        public Neuron(int numInputs, double LearnRate, int seed)
        {
            rnd = new Random(seed);

            learnRate = LearnRate;

            inputs = new double[numInputs];
            weights = new double[numInputs];

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = rnd.NextDouble(); ;//Each weight is random between -1 and 1
            }

            bias = rnd.NextDouble();//Random bias

            //Might be useful to be able to set the weights and biases here, or even the random seed
        }

        public static float Sigmoid(double value)//We love maths
        {
            return (float)(1.0 / (1.0 + Math.Pow(Math.E, -((value + 0) * 1))));
        }

        public static float TanH(double value)//Yes we do
        {
            return (float)((2 / (1 + Math.Exp(-2 * value))) - 1);
        }

        public void SetInputs(double[] input)//Guess what.
        {
            if (input.Length != weights.Length)
            {
                //Console.WriteLine("Wrong number of inputs, need " + weights.Count + " inputs");
                return;
            }
            
            //'What?'
            inputs = input;//The inputs are the inputs.
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
                total += inputs[i] * weights[i];//Add up the product for each connection of each neuron
            }

            output = Sigmoid(total - bias);//Squish that result

            return output;
        }

        public double CalcOutput(ActivationFunction activation)
        {
            double total = 0;

            if (inputs.Length != weights.Length)
            {
                //Console.WriteLine("Incorrect number of inputs, should be " + weights.Length + " inputs, there is " + inputs.Count + " inputs");
                return 0;
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                total += inputs[i] * weights[i];//Add up the product for each connection of each neuron
            }

            //Use correct activation function
            switch (activation)
            {
                default:
                    //Default to sigmoid
                    output = Sigmoid(total - bias);
                    break;
                case ActivationFunction.tanH:
                    output = TanH(total - bias);
                    break;
            }

            return output;
        }

        public double CalcOutputHidden()//Looks very similar, idk why this exists
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

        public void ReWeightOutput(double desiredOutput)//Broken
        {
            double error = desiredOutput - output;
            errorGradient = (long)(output * (1 - output) * error);

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * error;
            }

            bias += learnRate * -1 * errorGradient;
        }

        public void ReWeightOutput(double desiredOutput, double output)//Very broken
        {
            double error = desiredOutput - output;
            errorGradient = (long)(output * (1 - output) * error);

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * error;
            }

            bias += learnRate * -1 * errorGradient;
        }

        public void ReWeightHidden(Layer prevLayer, int weightNum)//Even more broken
        {
            errorGradient = (long)(output * (1 - output));
            double errorGradSum = 0;

            for (int i = 0; i < prevLayer.neurons.Length; i++)
            {
                errorGradSum += prevLayer.neurons[i].errorGradient * prevLayer.neurons[i].weights[weightNum];
            }

            errorGradient *= (long)errorGradSum;

            for (int i = 0; i < inputs.Length; i++)
            {
                weights[i] += learnRate * inputs[i] * errorGradient;
            }

            bias += learnRate * -1 * errorGradient;
        }
        
        public string GetWeightsString()//This works
        {
            string text = "";//Make a string of all the weights and bias

            foreach (double weight in weights)
            {
                text += weight.ToString("0.00000") + ",";//Need to do a little rounding, no one will notice
            }

            text += bias.ToString("0.00000");

            return text;//Because strings are cool
        }

        public void SetWeights(string weightsString)//We can get the weights and bias from a string
        {
            //Sould probably check that it is formatted right
            string[] arr = weightsString.Split(',');//Formatt is important

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Convert.ToDouble(arr[i]);//Set each weight
            }

            bias = Convert.ToDouble(arr[weights.Length]);//And set the bias
        }

        public void Mutate()//Because we like a genetic approach
        {
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] += weights[i] * 0.001f * rnd.Next(-100, 100);//Make random change by maximum of 10%
            }
            bias += bias * 0.001f * rnd.Next(-100, 100);//Make random change by maximum of 10%
        }

        public void Mutate(float effect)//Because we like a genetic approach
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += weights[i] * effect * 0.01f * rnd.Next(-100, 100);//Make random change by maximum of 10%
            }
            bias += bias * effect * 0.01f * rnd.Next(-100, 100);//Make random change by maximum of 10%
        }
    }
}
