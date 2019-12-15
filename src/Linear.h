#pragma once

#include <random>
#include <cmath>

#include "base_layers.h"


namespace MLComparison
{
	// class template for a standard linear neural network layer
	// with a given number of inputs and units (neurons)
	template<typename T, int n_inputs, int n_units>
	class Linear : public TrainableLayer<T, n_inputs, n_units>
	{
	public:

		// default constructor which initializes the input matrix pointer as
		// a null pointer and creates gradient matrices for the weights and biases
		Linear()
		{
		}


		// constructor which additionally sets the learning rate
		Linear(T learning_rate) : TrainableLayer<T, n_inputs, n_units>(learning_rate)
		{
		}

		
		// initializes weights according to the Kaiming He initialization scheme
		void kaiming_he_init()
		{
			// create random number generator
			std::default_random_engine rng;
			// normal distribution with mean 0 and sd of sqrt(2 / n_inputs)
			std::normal_distribution<T> dist_norm(0, std::sqrt(2.0 / n_inputs));
			// for each row of weights matrix
			for (int row = 0; row < n_inputs; row++)
			{
				// for each column of weights matrix
				for (int col = 0; col < n_units; col++)
				{
					// set weight to random number drawn from specified distribution
					weights[row][col] = dist_norm(rng);
				}
			}
		}


		// performs the forward pass, calculating outputs based on the inputs, weights and biases
		virtual Matrix<T, 1, n_units>* operator()(Matrix<T, 1, n_inputs>* x) override
		{
			// save pointer to the input matrix so its gradients can be set during the backward pass
			forward_record.input_matrix = x;
			// output matrix is the biases added to the dot product of input matrix and weights, 
			// i.e the output of each unit is the sum of each input multiplied by that unit's corresponding
			// weight parameter, plus the bias terms
			forward_record.output_matrix = x->dot(weights).add(biases);
			// return the outputs
			return &forward_record.output_matrix;
		}


		// performs the backwards pass, calculating the gradients of the input matrix and the parameters
		// based on the gradients of the output matrix and the derivative of the current function
		virtual void backward() override
		{
			// if input matrix exists
			if (forward_record.input_matrix != nullptr)
			{
				// gradients of the input matrix are the dot product of the gradients of the output matrix and the
				// transpose of the weights, i.e the gradient of an input element is the sum of the elementwise
				// product of the gradients of each neuron's output and the weights of each neuron for that input
				if (auto* ptr = dynamic_cast<GradMatrix<T, 1, n_inputs>*>(forward_record.input_matrix))
				{
					ptr->grad = forward_record.output_matrix.grad.dot_t(weights);
				}
				// gradients of the weights are the dot product of the transpose of the input matrix and the gradients
				// of the output matrix, i.e. the gradient of each weight is the product of the input corresponding to
				// that weight and the gradient of the output of the unit to which that weight belongs
				weights.grad = forward_record.input_matrix->t_dot(forward_record.output_matrix.grad);
				// gradients of the biases are simply the gradients of the outputs
				biases.grad = forward_record.output_matrix.grad;
			}
		}

		
		// updates the weights and biases
		virtual void update() override
		{
			weights.SGDStep(this->learning_rate);
			biases.SGDStep(this->learning_rate);
		}


	private:

		// record of forward pass
		ForwardRecord<T, n_inputs, n_units> forward_record;

		// matrix of weights
		GradMatrix<T, n_inputs, n_units> weights;
		// matrix of biases
		GradMatrix<T, 1, n_units> biases;
	};
}