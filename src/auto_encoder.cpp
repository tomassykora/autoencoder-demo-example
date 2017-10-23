/* 
 * Tomas Sykora,
 * tms.sykora@gmail.com,
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <cmath>
#include <random>
#include <chrono>

typedef std::vector<std::vector<double>> Matrix2D;

Matrix2D matrixMul(Matrix2D m1, Matrix2D m2)
{
	if (m1[0].size() != m2.size())
	{
		std::cerr << "Error: Wrong format of matrices: [" << m1.size() << ", " << m1.back().size() << "] "
		                                         << "x [" << m2.size() << ", " << m2.back().size() << "]" << std::endl;
	}

	Matrix2D output_matrix;

	for (int r = 0; r < m1.size(); ++r)
	{
		std::vector<double> new_row;
		for (int c = 0; c < m2.back().size(); ++c)
		{
			double sum = 0;
			for (int i = 0; i < m1.back().size(); ++i)
			{
				sum += m1[r][i] * m2[i][c];
			}
			new_row.push_back(sum);
		}
		output_matrix.push_back(new_row);
	}

	return output_matrix;
}


class Layer 
{
	// One layer consists of two weight matrices and of
	// two vectors for deltas. It's to compute the decoder
	// output of the auto-encoder ( - a NN with one hidden 
	// layer and one output layer and its output is trained
	// to be the same as the input, which is used as pretraining
	// of the weights). 

public:
	Layer(unsigned thisL, unsigned nextL);

	double getWeight(unsigned i, unsigned j) { return W[i][j]; }
	double getDecoderWeight(unsigned i, unsigned j) { return W_decode[i][j]; }

	void updateWeight(double d_w, unsigned i, unsigned j) { W[i][j] += d_w; }
	void updateDecoderWeight(double d_w, unsigned i, unsigned j) { W_decode[i][j] += d_w; }

	Matrix2D getWeights() { return W; }
	Matrix2D getDecoderWeights() { return W_decode; }

	double getDelta(unsigned index) { return deltas[index]; }
	double getDecoderDelta(unsigned index) { return deltas_decode[index]; }

	void setDelta(double value, int index) { deltas[index] = value; }
	void setDecoderDelta(double value, int index) { deltas_decode[index] = value; }

	void setOutputs(Matrix2D y) { outs = y;}
	Matrix2D getOutputs() { return outs; }

	double getOutput(unsigned iter, unsigned index) { return outs[iter][index]; }

	unsigned size() { return deltas.size(); }

private:
	Matrix2D W;
	Matrix2D W_decode;
	std::vector<double> deltas;
	std::vector<double> deltas_decode;
	Matrix2D outs;
};

Layer::Layer(unsigned thisL, unsigned nextL)
{
	// Weights initialization to random interval of (-0.5, 0.5)

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator (seed);
	std::uniform_real_distribution<double> distribution(-0.5, 0.5);

	// weight matrix W and a vector of neurons deltas:
	for (int i = 0; i < thisL + 1; ++i) // (thisL + 1) for bias
	{
		std::vector<double> w_row;
		for (int j = 0; j < nextL; ++j)
		{
			double val = distribution(generator);
			w_row.push_back(val);
		}
		W.push_back(w_row);
		if (i != thisL) // don't want to compute a delta for bias
		{
			deltas.push_back(0.0);
			deltas_decode.push_back(0.0);
		}
	}

	// A decoder/output layer of the auto-encoder:
	for (int i = 0; i < nextL + 1; ++i)
	{
		std::vector<double> row;
		for (int j = 0; j < thisL; ++j)
		{
			row.push_back(distribution(generator));
		}
		W_decode.push_back(row);
	}
}

class NNetwork 
{
public:
	NNetwork(const std::vector<unsigned> topology);
	void autoEncode(Matrix2D x);
private:
	Matrix2D sigmoid(Matrix2D x);
	Matrix2D normalize(Matrix2D x);

	void backPropagate(unsigned index);
	Matrix2D feedForward(unsigned index);

	std::vector<Layer> layers;
	std::vector<unsigned> topology;
	Matrix2D input;

	double training_step;
};

NNetwork::NNetwork(const std::vector<unsigned> top)
{
	std::cout << "\n ++ Creating a NN with a topology: ( ";

	topology = top;

	for (std::vector<unsigned>::const_iterator i = top.begin(); i != top.end(); ++i)
	{
		std::cout << *i << " ";
	}
	std::cout << ") ++" << std::endl;

	training_step = 0.1;

	std::cout << "\n Set value for training step): ";
	std::cin >> training_step; // doesn't check the correct input as it's just a demo app

	// Create required layers:
	for (int layerNum = 0; layerNum < topology.size(); ++layerNum)
	{	
		unsigned nextL = (layerNum == topology.size() - 1) ? 1 : topology[layerNum + 1];
		layers.push_back(Layer(topology[layerNum], nextL));
	}
}

Matrix2D NNetwork::normalize(Matrix2D x)
{
	// Normalize all input values to the interval of (0, 1)

	Matrix2D x_normalized;

	// Find minimum and maximum values in each in put class:
	std::vector<double> maxs = x[0];
	std::vector<double> mins = x[0];
	for (int col = 0; col < x.back().size(); ++col)
	{
		for (int row = 0; row < x.size(); ++row)
		{
			maxs[col] = std::max(x[row][col], maxs[col]);
			mins[col] = std::min(x[row][col], mins[col]);
		}
	}

	for (int i = 0; i < x.size(); ++i)
	{
		std::vector<double> row;
		for (int j = 0; j < x.back().size(); ++j)
		{
			// Normalization:
			double val = (x[i][j] - mins[j]) / (maxs[j] - mins[j]);
			row.push_back(val);
		}
		x_normalized.push_back(row);
	}

	return x_normalized;
}

Matrix2D NNetwork::sigmoid(Matrix2D x)
{
	// Compute sigmoid from all matrix values

	Matrix2D sigm_out;

	for (int i = 0; i < x.size(); ++i)
	{
		std::vector<double> row;

		for (int j = 0; j < x.back().size(); ++j)
		{
			double val = 1 / (1 + exp(-x[i][j]));
			row.push_back(val);
		}

		sigm_out.push_back(row);
	}

	return sigm_out;
}

Matrix2D NNetwork::feedForward(unsigned index_l)
{
	// Feed through the hidden layer (the layer we're pretraining its weights):

	Matrix2D w = layers[index_l].getWeights();
	Matrix2D x;

	index_l == 0 ? x = input
			   : x = layers[index_l].getOutputs();

	for (int i = 0; i < x.size(); ++i) // add bias to every training sample
	{
		x[i].push_back(1.0);
	}

	Matrix2D y_hidden = sigmoid(matrixMul(x, w));
	layers[index_l + 1].setOutputs(y_hidden);

	// Feed through the decoder/output layer, where same (trained to output the values from the input):
	for (int i = 0; i < y_hidden.size(); ++i) // add bias
	{
		y_hidden[i].push_back(1.0);
	}
	Matrix2D w_decode = layers[index_l].getDecoderWeights();
	Matrix2D y_decoded = sigmoid(matrixMul(y_hidden, w_decode));

	return y_decoded;
}


/***
	 Backpropagation through one hidden layer and one decoder/output layer (not deep
   NNs, as we're building an autoencoder which trains just one layer at a time). 
   unsigned index; - index of the currently trained network layer. 

***/

void NNetwork::backPropagate(unsigned index)
{
	std::cout << "\n   An example output with random weights (before training): \n";

	std::vector<double> in = (index == 0) ? input[0]
										 : layers[index].getOutputs()[0];
	Matrix2D out = feedForward(index);

	std::cout << "input = [ ";
	for (int i = 0; i < in.size(); ++i)
	{
		std::cout << in[i] << " ";
	}
	std::cout << "]\n";
	std::cout << "output = [ ";
	for (int i = 0; i < out.back().size(); ++i)
	{
		std::cout << out[0][i] << " ";
	}
	std::cout << "]\n";


	// Traning

	std::cout << "\n   5000 epochs of training started.\n";

	int epoch = 0;
	double Err;

	do {

		Err = 0;
		
		Matrix2D y = feedForward(index);

		for (int d = 0; d < y.size(); ++d) // through the whole training set
		{	
			// Input of the first network layer is the original input,
			// input of other layers is output of the previous layer
			std::vector<double> x = (index == 0) ? input[d]
											 : layers[index].getOutputs()[d];

			double Ep = 0; // Mean squared error (the output is traned to be the same as the input)
			               // (input == ground truth)

			// Through the output layer neurons:
			for (int i = 0; i < y[d].size(); ++i) 
			{
				Ep += 0.5 * pow((x[i] - y[d][i]), 2);
				layers[index].setDecoderDelta((x[i] - y[d][i]) * y[d][i] * (1 - y[d][i]), i);
			}

			Err += Ep;
			
			// Through the hidden layer neurons:
			for (int i = 0; i < layers[index + 1].size(); ++i) 
			{
				double sum = 0;
				for (int j = 0; j < layers[index].size(); ++j) // Sum of the output layer deltas * weight (sum throuh deltas)
				{
					sum += layers[index].getDecoderDelta(j) * layers[index].getDecoderWeight(i, j);
				}

				layers[index + 1].setDelta(sum * layers[index + 1].getOutput(d, i) * (1 - layers[index + 1].getOutput(d, i)), i);
			}

			// Gradient Descent:
			x.push_back(1.0); // bias
			
			// Update weights of the layer
			for (int i = 0; i < layers[index + 1].size(); ++i) // through deltas of the next layer
			{
				for (int j = 0; j < x.size(); ++j) // through input x
				{
					double d_w = training_step * layers[index + 1].getDelta(i) * x[j];
					layers[index].updateWeight(d_w, j, i);
				}
			}

			// Updates decoder layer weights
			for (int i = 0; i < layers[index].size(); ++i) // through deltas of the helper decoder layer
			{
				for (int j = 0; j < layers[index + 1].size(); ++j) // through outputs of the hidden layer
				{
					double d_w = training_step * layers[index].getDecoderDelta(i) * layers[index + 1].getOutput(d, j);
					layers[index].updateDecoderWeight(d_w, j, i);
				}
			}	

		} // for(training set) 

		std::cout << "\r    EPOCH: " << epoch << " -- error: " << Err;

		epoch++;

	} while (epoch < 5000);

	std::cerr << std::endl;
	std::cout << "\n  The layer " << index << " was trained. An example output given the input data: \n\n";

	std::vector<double> x = (index == 0) ? input[0]
										 : layers[index].getOutputs()[0];
	Matrix2D y = feedForward(index);

	std::cout << "input = [ ";
	for (int i = 0; i < x.size(); ++i)
	{
		std::cout << x[i] << " ";
	}
	std::cout << "]\n";
	std::cout << "output = [ ";
	for (int i = 0; i < y.back().size(); ++i)
	{
		std::cout << y[0][i] << " ";
	}
	std::cout << "]\n";
}

void NNetwork::autoEncode(Matrix2D x)
{
	input = normalize(x);

	std::cout << "\nInput data (x) having format: [" << x.size() << ", " << x.back().size() << "]\n";

	std::cout << "\n * * * Backpropagation : \n";

	for (int layer = 0; layer < topology.size() - 1; ++layer)
	{
		std::cout << "\n  * Starting with backpropagation of the layer " << layer << std::endl;
		backPropagate(layer);
	}
}

Matrix2D makeUniformData(unsigned n, unsigned size)
{
	// Function produces 'size' numbers for each of 'n' classes of a uniform 
	// distribution from a random range

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator (seed);

	Matrix2D x;

	for (int i = 0; i < size; ++i)
	{
		std::vector<double> row;
		for (int j = 0; j < n; ++j)
		{
			std::uniform_real_distribution<double> distribution(j*5, j*5+10); // just some numbers to generate for demonstration ...
    		row.push_back(distribution(generator));
		}
		x.push_back(row);	
	}

	return x;
}

Matrix2D readData()
{
	Matrix2D data;
  	std::ifstream infile( "/home/tosykora/Documents/SFC/autoencoder-demo-example/train.csv" );

	while (infile)
	{
		std::string s;
		if (!getline( infile, s )) break;

		std::istringstream ss( s );
		std::vector <double> record;

		bool first_column = true; // get rid of the first ID column
		while (ss)
		{
		  std::string s;
		  if (!getline( ss, s, ',' )) break;

		  double x;
		  std::istringstream xx(s);
		  xx >> x;

		  if (!first_column)
		  	record.push_back( x );
		  first_column = false;
		}

		data.push_back( record );
	}

	if (!infile.eof())
	{
		std::cerr << "Error reading the file!\n";
	}

	return data;
}



int main(int argc, char const *argv[])
{
	// A simple demo showing a basic auto-encoder pretraining the weights 
	// of a neural network with a given topology on randomly generated input,
	// or a demonstration of a NN traning on .csv dataset (e.g. Boston houses).
	
	Matrix2D x; // To store the input vector

	std::vector<unsigned> topology;

	if (argc == 1)
	{
		std::cout << "\n Reading training data from the .csv dataset file...\n";

		static const unsigned arr[] = {14, 7, 5, 2};
		const std::vector<unsigned> vec (arr, arr + sizeof(arr) / sizeof(arr[0]) );

		topology = vec;

		// Read training data from a csv
		x = readData();
	}
	else if (argc > 1)
	{
		std::cout << "\n Creating a training set (random uniform distributions)...\n";

		// Load topology from cmd parameters

		for (int i = 1; i < argc; ++i)
		{
			std::istringstream ss(argv[i]);
			int p;
			if (!(ss >> p) || p < 1)
			{
	    		std::cerr << "Error: Invalid argument (must be integer above 0): " << argv[i] << std::endl;

	    		return 1;
			}
	    	else
	    	{
	    		topology.push_back(p);
	    	}
		}

		// Create random training data
		x = makeUniformData(topology[0], 1000);
	}
	else 
	{
		std::cerr << "Error: Must set at least 2 integer parameters or none.\n";

		return 1;
	}


	NNetwork network(topology);
	network.autoEncode(x);

	return 0;
}















