using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public class Layer
    {
        private int nInputs;
        private double[] inputs;
        private double[] outputs;
        public Neuron[] neurons;

        public Layer(int numNeurons, int numInputs, double learnRate)
        {
            nInputs = numInputs;
            neurons = new Neuron[numNeurons];

            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(numInputs, learnRate);
            }

            outputs = new double[numNeurons];
        }

        public void SetInputs(double[] input)
        {
            if (input.Length != nInputs)
            {
                //Console.WriteLine("Incorrect number of inputs for this layer, should be " + nInputs + " inputs, there is " + input.Count + " inputs");
                return;
            }

            inputs = input;

            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i].SetInputs(inputs);
            }
        }

        public double[] CalcOutput()
        {
            for(int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = neurons[i].CalcOutput();
            }

            return outputs;
        }

        public double[] CalcOutputHidden()
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                outputs[i] = neurons[i].CalcOutputHidden();
            }

            return outputs;
        }

        public void ReWeightOutput(double[] desiredOutput, double[] output)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].ReWeightOutput(desiredOutput[i], output[i]);
            }
        }

        public void ReWeightHidden(Layer prevLayer)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].ReWeightHidden(prevLayer, i);
            }
        }

        public void Mutate()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].Mutate();
            }
        }
    }
}
