using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MyAI
{
    public class NeuralNetwork
    {
        private Layer[] layers;//The layers of the network
        private double[][] arrOfOutputs;//Array to keep track of the outputs from each layer during output calculation, mainly for debugging
        
        //The various sizes of the network
        public readonly int numOfHidden;
        public readonly int numOfInputs;
        public readonly int layerSize;
        public readonly int numOfOuputs;
        //The various sizes of the network

        public readonly double learnRate = 0.1;

        private ActivationFunction activationFunction;

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
                arrOfOutputs = new double[numHiddenLayers + 2][];//Keep track of all outputs
            }
            else
            {
                arrOfOutputs = new double[numHiddenLayers + 2][];//Keep track of all outputs
            }

            layers = new Layer[numHiddenLayers + 2];//The input layer, the hidden layers, and the output layer

            layers[0] = new Layer(sizeOfLayers, numInputs, learnRate, -1);//Input layer

            for (int i = 0; i < numHiddenLayers; i++)
            {
                layers[i + 1] = new Layer(sizeOfLayers, sizeOfLayers, learnRate, i);//Hidden layers
            }

            layers[numHiddenLayers + 1] = new Layer(numOutputs, sizeOfLayers, learnRate, numHiddenLayers + 1);//Output layer

            activationFunction = ActivationFunction.sigmoid;

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

        public NeuralNetwork(int numHiddenLayers, int numInputs, int numOutputs, int sizeOfLayers, ActivationFunction function) : this(numHiddenLayers, numInputs, numOutputs, sizeOfLayers)
        {
            activationFunction = function;
        }

        public void SetInputs(double[] input)
        {
            layers[0].SetInputs(input);//Set the inputs for the input layer
        }

        public double[] CalcOutput()
        {
            double[] lastOutput;

            lastOutput = layers[0].CalcOutput(activationFunction);//Output from input layer

            arrOfOutputs[0] = lastOutput;//Store the outputs from each layer for debugging

            for (int i = 1; i < layers.Length; i++)
            {
                arrOfOutputs[i] = lastOutput;//Store the outputs from each layer for debugging

                layers[i].SetInputs(lastOutput);//Set inputs of hidden layer to output of last layer
                lastOutput = layers[i].CalcOutput(activationFunction);//Get the output
            }

            return lastOutput;//Return the output of the last layer
        }

        public void Correct(double[] desiredOuput, double[] output)//Broken, dont use
        {
            layers[layers.Length - 1].ReWeightOutput(desiredOuput, output);//Reweight the output layer

            for (int i = layers.Length - 2; i >= 0; i--)
            {
                layers[i].ReWeightHidden(layers[i + 1]);//Reweight each hidden layer
            }
        }

        public void Train(double[][] inputs, double[][] desiredOutputs)//Broken due to Correct() being broken
        {
            var usedNums = new List<int>();
            var rnd = new Random();
            int i = 0;
            while(usedNums.Count != inputs.Length)
            {
                while (usedNums.Contains(i))
                {
                    i = rnd.Next(0, inputs.Length);
                }
                usedNums.Add(i);

                double[] thisInput = inputs[i];

                SetInputs(thisInput);//Set a new input
                double[] output = CalcOutput();//Calculate the new output

                double[] thisDesired = desiredOutputs[i];

                Correct(thisDesired, output);//Correct for any mistakes
            }
        }

        public string[] GetWeights()//So the current weights can be loaded into another network
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

        public void SaveWeights(string filePath)//Saves the weights for later use
        {
            string[] weightText = GetWeights();
            //Maybe use some compression algoithm
            File.WriteAllLines(filePath, weightText);
        }

        public void LoadWeights(string[] weightsString)//Sets the weights from another network
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

        public void LoadFileWeights(string filePath)//Can get weights from a file to save training again
        {
            string[] lines = File.ReadAllLines(filePath);
            //If compressed needs to be decompressed
            LoadWeights(lines);
        }

        public void Mutate()//Randomly changes everything by a small amount
        {
            for(int i = 0; i < layers.Length; i++)
            {
                layers[i].Mutate();
            }
        }

        public void Mutate(float effect)//Randomly changes everything by a small amount
        {
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].Mutate(effect);
            }
        }

        public double[][] GetOuputs()
        {
            return arrOfOutputs;
        }
    }
}

