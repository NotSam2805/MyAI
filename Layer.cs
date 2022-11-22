using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public class Layer
    {
        private readonly int nInputs;//The size of input arrays that is expected
        
        //The inputs and outputs of this layer
        private double[] inputs;
        private double[] outputs;
        
        public Neuron[] neurons;//Cant remember why this is public, it feels like it shouldn't be

        public Layer(int numNeurons, int numInputs, double learnRate)
        {
            //Set the number of inputs and the size of the array
            nInputs = numInputs;
            neurons = new Neuron[numNeurons];

            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(numInputs, learnRate);//Make each neuron
            }

            outputs = new double[numNeurons];//Don't know why I didn't put this at the top with the others, I wonder if it's lonely.
        }

        public void SetInputs(double[] input)
        {
            if (input.Length != nInputs)//Checks that the input is the right formatt
            {
                //Console.WriteLine("Incorrect number of inputs for this layer, should be " + nInputs + " inputs, there is " + input.Count + " inputs");
                return;
            }

            inputs = input;//Save this for debugging

            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i].SetInputs(inputs);//Set the input of each neuron
            }
        }

        public double[] CalcOutput()
        {
            for(int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = neurons[i].CalcOutput();//Get the output of each neuron
            }

            return outputs;
        }

        public double[] CalcOutputHidden()//Ignore this, I was trying something
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = neurons[i].CalcOutputHidden();
            }

            return outputs;
        }

        public void ReWeightOutput(double[] desiredOutput, double[] output)//Broken
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].ReWeightOutput(desiredOutput[i], output[i]);
            }
        }

        public void ReWeightHidden(Layer prevLayer)//Very broken
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].ReWeightHidden(prevLayer, i);
            }
        }

        public void Mutate()//Read comment in NeuralNetwork.cs
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].Mutate();
            }
        }
    }
}
