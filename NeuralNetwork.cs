using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MyAI
{
    public class NeuralNetwork
    {
        private List<Layer> layers;
        private List<double>[] arrOfOutputs;
        public int numOfHidden;
        public int numOfInputs;
        public int layerSize;
        public int numOfOuputs;

        public NeuralNetwork(int numHiddenLayers, int numInputs, int numOutputs, int sizeOfLayers)
        {
            numOfHidden = numHiddenLayers;
            numOfInputs = numInputs;
            layerSize = sizeOfLayers;
            numOfOuputs = numOutputs;

            arrOfOutputs = new List<double>[numHiddenLayers + 2];//Keep track of all outputs

            layers = new List<Layer>();

            layers.Add(new Layer(sizeOfLayers, numInputs, 0.5));//input layer

            for (int i = 0; i < numHiddenLayers; i++)
            {
                layers.Add(new Layer(sizeOfLayers, sizeOfLayers, 0.5));//hidden layers
            }

            layers.Add(new Layer(numOutputs, sizeOfLayers, 0.5));//output layer

            List<double> defaultInputs = new List<double>();
            for(int i = 0; i < numInputs; i++)
            {
                defaultInputs.Add(0);
            }

            SetInputs(defaultInputs);
        }

        public void SetInputs(List<double> input)
        {
            layers[0].SetInputs(input);//set the inputs for the input layer
        }

        public List<double> CalcOutput()
        {
            List<double> lastOutput;

            lastOutput = layers[0].CalcOutput();//output from input layer
            arrOfOutputs[0] = lastOutput;//adds to outputs

            for (int i = 1; i < layers.Count; i++)
            {
                layers[i].SetInputs(lastOutput);//set inputs of hidden layer to output of last layer
                lastOutput = layers[i].CalcOutput();//get the output
                arrOfOutputs[i] = lastOutput;//keep track of outputs
            }

            return lastOutput;//return the output of the last layer
        }

        public void Correct(List<double> desiredOuput, List<double> output)
        {
            layers[layers.Count - 1].ReWeightOutput(desiredOuput, output);//reweight the output layer

            for (int i = layers.Count - 2; i >= 0; i--)
            {
                layers[i].ReWeightHidden(layers[i + 1]);//reweight each hidden layer
            }
        }

        public void Train(List<double>[] inputs, List<double>[] desiredOutputs)
        {
            for (int i = 0; i < inputs.Length; i++)//for every list of input
            {
                SetInputs(inputs[i]);//set a new input
                List<double> output = CalcOutput();//calculate the new output
                Correct(desiredOutputs[i], output);//correct for any mistakes
            }
        }

        public string[] GetWeights()
        {
            List<string> weights = new List<string>();

            for(int i = 0; i < layers.Count; i++)
            {
                weights.Add(i.ToString());
                foreach(Neuron neuron in layers[i].neurons)
                {
                    weights.Add(neuron.GetWeightsString());
                }
            }

            return weights.ToArray();
        }

        public void SaveWeights(string filePath)
        {
            string[] weightText = GetWeights();

            File.WriteAllLines(filePath, weightText);
        }

        public void LoadWeights(string[] weightsString)
        {
            int layerNum = 0;
            int neuronCount = 0;
            for (int a = 0; a < weightsString.Length; a++)
            {
                if (!weightsString[a].Contains(","))
                {
                    layerNum = Convert.ToInt32(weightsString[a]);
                    neuronCount = 0;
                }
                else
                {
                    layers[layerNum].neurons[neuronCount].SetWeights(weightsString[a]);
                    neuronCount++;
                }
            }
        }

        public void LoadFileWeights(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);
            LoadWeights(lines);
        }

        public void Mutate()
        {
            for(int i = 0; i < layers.Count; i++)
            {
                layers[i].Mutate();
            }
        }
    }
}
