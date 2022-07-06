using System;
using System.Collections.Generic;
using System.Text;

namespace MyAI
{
    public class Layer
    {
        private int nInputs;
        private List<double> inputs;
        private List<double> outputs;
        public List<Neuron> neurons;

        public Layer(int numNeurons, int numInputs, double learnRate)
        {
            nInputs = numInputs;
            neurons = new List<Neuron>();

            outputs = new List<double>();

            for (int i = 0; i < numNeurons; i++)
            {
                neurons.Add(new Neuron(numInputs, learnRate));
            }
        }

        public void SetInputs(List<double> input)
        {
            if (input.Count != nInputs)
            {
                //Console.WriteLine("Incorrect number of inputs for this layer, should be " + nInputs + " inputs, there is " + input.Count + " inputs");
                return;
            }

            inputs = input;

            foreach (Neuron neuron in neurons)
            {
                neuron.SetInputs(inputs);
            }
        }

        public List<double> CalcOutput()
        {
            outputs.Clear();

            foreach (Neuron neuron in neurons)
            {
                outputs.Add(neuron.CalcOutput());
            }

            return outputs;
        }

        public void ReWeightOutput(List<double> desiredOutput, List<double> output)
        {
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].ReWeightOutput(desiredOutput[i], output[i]);
            }
        }

        public void ReWeightHidden(Layer prevLayer)
        {
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].ReWeightHidden(prevLayer, i);
            }
        }

        public void Mutate()
        {
            for (int i = 0; i < neurons.Count; i++)
            {
                neurons[i].Mutate();
            }
        }
    }
}
