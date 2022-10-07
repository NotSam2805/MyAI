using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MyAI
{
    public class NeuralNetwork
    {
        private Layer[] layers;
        private double[,] arrOfOutputs;
        public readonly int numOfHidden;
        public readonly int numOfInputs;
        public readonly int layerSize;
        public readonly int numOfOuputs;

        public NeuralNetwork(int numHiddenLayers, int numInputs, int numOutputs, int sizeOfLayers)
        {
            numOfHidden = numHiddenLayers;
            numOfInputs = numInputs;
            layerSize = sizeOfLayers;
            numOfOuputs = numOutputs;

            if(sizeOfLayers > numOfOuputs)
            {
                arrOfOutputs = new double[numHiddenLayers + 2, sizeOfLayers];//Keep track of all outputs
            }
            else
            {
                arrOfOutputs = new double[numHiddenLayers + 2, numOfOuputs];//Keep track of all outputs
            }

            layers = new Layer[numHiddenLayers + 2];

            layers[0] = new Layer(sizeOfLayers, numInputs, 0.5);//input layer

            for (int i = 0; i < numHiddenLayers; i++)
            {
                layers[i + 1] = new Layer(sizeOfLayers, sizeOfLayers, 0.5);//hidden layers
            }

            layers[numHiddenLayers + 1] = new Layer(numOutputs, sizeOfLayers, 0.5);//output layer

            /*
            List<double> defaultInputs = new List<double>();
            for(int i = 0; i < numInputs; i++)
            {
                defaultInputs.Add(0);
            }

            SetInputs(defaultInputs);
            */
        }

        public void SetInputs(double[] input)
        {
            layers[0].SetInputs(input);//set the inputs for the input layer
        }

        public double[] CalcOutput()
        {
            double[] lastOutput;

            lastOutput = layers[0].CalcOutput();//output from input layer

            for (int i = 1; i < layers.Length; i++)
            {
                for (int a = 0; a < lastOutput.Length; a++)
                {
                    arrOfOutputs[i, a] = lastOutput[a];//keep track of outputs
                }

                layers[i].SetInputs(lastOutput);//set inputs of hidden layer to output of last layer
                lastOutput = layers[i].CalcOutput();//get the output
            }

            for (int a = 0; a < lastOutput.Length; a++)
            {
                arrOfOutputs[layers.Length - 1, a] = lastOutput[a];//keep track of outputs
            }

            return lastOutput;//return the output of the last layer
        }

        public void Correct(double[] desiredOuput, double[] output)
        {
            layers[layers.Length - 1].ReWeightOutput(desiredOuput, output);//reweight the output layer

            for (int i = layers.Length - 2; i >= 0; i--)
            {
                layers[i].ReWeightHidden(layers[i + 1]);//reweight each hidden layer
            }
        }

        public void Train(double[,] inputs, double[,] desiredOutputs)
        {
            for (int i = 0; i < inputs.GetLength(0); i++)//for every list of input
            {
                double[] thisInput = new double[numOfInputs];
                for(int a = 0; a < thisInput.Length; a++)
                {
                    thisInput[a] = inputs[i,a];
                }

                SetInputs(thisInput);//set a new input
                double[] output = CalcOutput();//calculate the new output

                double[] thisDesired = new double[numOfOuputs];
                for (int a = 0; a < thisDesired.Length; a++)
                {
                    thisDesired[a] = desiredOutputs[i, a];
                }

                Correct(thisDesired, output);//correct for any mistakes
            }
        }

        public string[] GetWeights()//so the current weights can be loaded into another network
        {
            List<string> weights = new List<string>();

            for(int i = 0; i < layers.Length; i++)
            {
                weights.Add(i.ToString());
                foreach(Neuron neuron in layers[i].neurons)
                {
                    weights.Add(neuron.GetWeightsString());
                }
            }

            return weights.ToArray();
        }

        public void SaveWeights(string filePath)//saves the weights for later use
        {
            string[] weightText = GetWeights();

            File.WriteAllLines(filePath, weightText);
        }

        public void LoadWeights(string[] weightsString)//sets the weights from another network
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

        public void LoadFileWeights(string filePath)//can get weights from a file to save training again
        {
            string[] lines = File.ReadAllLines(filePath);
            LoadWeights(lines);
        }

        public void Mutate()//randomly changes everything by a small amount
        {
            for(int i = 0; i < layers.Length; i++)
            {
                layers[i].Mutate();
            }
        }

        public double[,] GetOuputs()
        {
            return arrOfOutputs;
        }
    }
}

