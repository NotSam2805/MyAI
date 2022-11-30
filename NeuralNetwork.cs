using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MyAI
{
    public class NeuralNetwork
    {
        private Layer[] layers;//The layers of the network
        private double[][] arrOfOutputs;//array to keep track of the outputs from each layer during output calculation, mainly for debugging
        
        //The various sizes of the network
        public readonly int numOfHidden;
        public readonly int numOfInputs;
        public readonly int layerSize;
        public readonly int numOfOuputs;
        //The various sizes of the network

        public NeuralNetwork(int numHiddenLayers, int numInputs, int numOutputs, int sizeOfLayers)
        {
            //Set all the sizes
            numOfHidden = numHiddenLayers;
            numOfInputs = numInputs;
            layerSize = sizeOfLayers;
            numOfOuputs = numOutputs;
            //Set all the sizes

            //The array of outputs needs to be the right size, either the size of the layers or the number of outputs (whichever is larger)
            if(sizeOfLayers > numOfOuputs)
            {
                arrOfOutputs = new double[numHiddenLayers + 2, sizeOfLayers];//Keep track of all outputs
            }
            else
            {
                arrOfOutputs = new double[numHiddenLayers + 2, numOfOuputs];//Keep track of all outputs
            }

            layers = new Layer[numHiddenLayers + 2];//The input layer, the hidden layers, and the output layer

            layers[0] = new Layer(sizeOfLayers, numInputs, 0.5, -1);//input layer

            for (int i = 0; i < numHiddenLayers; i++)
            {
                layers[i + 1] = new Layer(sizeOfLayers, sizeOfLayers, 0.5, i);//hidden layers
            }

            layers[numHiddenLayers + 1] = new Layer(numOutputs, sizeOfLayers, 0.5, numHiddenLayers + 1);//output layer

            /*
            //Used for debugging
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
                arrOfOutputs = lastOutput;

                layers[i].SetInputs(lastOutput);//set inputs of hidden layer to output of last layer
                lastOutput = layers[i].CalcOutput();//get the output
            }

            arrOfOutputs = lastOutput;

            return lastOutput;//return the output of the last layer
        }

        public void Correct(double[] desiredOuput, double[] output)//Broken, dont use
        {
            layers[layers.Length - 1].ReWeightOutput(desiredOuput, output);//reweight the output layer

            for (int i = layers.Length - 2; i >= 0; i--)
            {
                layers[i].ReWeightHidden(layers[i + 1]);//reweight each hidden layer
            }
        }

        public void Train(double[][] inputs, double[][] desiredOutputs)//Broken due to Correct() being broken
        {
            for (int i = 0; i < inputs.GetLength(0); i++)//for every list of input
            {
                double[] thisInput = inputs[i];

                SetInputs(thisInput);//set a new input
                double[] output = CalcOutput();//calculate the new output

                double[] thisDesired = desiredOutputs[i];

                Correct(thisDesired, output);//correct for any mistakes
            }
        }

        public string[] GetWeights()//so the current weights can be loaded into another network
        {
            List<string> weights = new List<string>();//An array might be better but a list is easier to use

            for(int i = 0; i < layers.Length; i++)
            {
                weights.Add(i.ToString());//The layer 'number'
                foreach(Neuron neuron in layers[i].neurons)
                {
                    weights.Add(neuron.GetWeightsString());//Get the weights of each neuron
                }
            }

            return weights.ToArray();
        }

        public void SaveWeights(string filePath)//saves the weights for later use
        {
            string[] weightText = GetWeights();
            //Maybe use some compression algoithm
            File.WriteAllLines(filePath, weightText);
        }

        public void LoadWeights(string[] weightsString)//sets the weights from another network
        {
            //Should add some check that the string array is for the right number of neurons
            //Should also check the formatt of the string array
            
            int layerNum = 0;
            int neuronCount = 0;
            for (int a = 0; a < weightsString.Length; a++)//foreach loop might be easier to read?
            {
                if (!weightsString[a].Contains(","))//The only line with no comma is the layer number
                {
                    layerNum = Convert.ToInt32(weightsString[a]);
                    neuronCount = 0;
                }
                else
                {
                    layers[layerNum].neurons[neuronCount].SetWeights(weightsString[a]);//Set the weights of the neuron
                    neuronCount++;//To the next neuron
                }
            }
        }

        public void LoadFileWeights(string filePath)//can get weights from a file to save training again
        {
            string[] lines = File.ReadAllLines(filePath);
            //if compressed needs to be decompressed
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

