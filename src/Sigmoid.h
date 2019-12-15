#pragma once

#include <cmath>

#include "base_layers.h"


namespace MLComparison
{
	// class template for a sigmoid activation function layer whose inputs are of the given dimensions
	template<typename T, int cols>
	class Sigmoid : public Layer<T, cols, cols>
	{
	public:

		// default constructor
		Sigmoid()
		{
		}


		// forward pass which applies the sigmoid function to each input element
		virtual Matrix<T, 1, cols>* operator()(Matrix<T, 1, cols>* x) override
		{
			// set pointer to address of matrix given as input
			forward_record.input_matrix = x;
			// for each column of input/output matrices
			for (int col = 0; col < cols; col++)
			{
				// set element of output matrix to current item if positive, else 0
				forward_record.output_matrix[0][col] = sigmoid(x->at(0)[col]);
			}
			// return reference to output matrix
			return &forward_record.output_matrix;
		}

		
		// backward pass which sets gradient of each input based on gradients of outputs
		// and derivative of sigmoid function
		virtual void backward() override
		{
			// GradMatrix pointer to input matrix
			auto* grad_input_ptr = static_cast<GradMatrix<T, 1, cols>*>(forward_record.input_matrix);
			// temporary variable for current element of input matrix
			T current_elem;
			// for each column of input.output matrices
			for (int col = 0; col < cols; col++)
			{
				current_elem = forward_record.input_matrix->at(0)[col];
				// gradient of input is derivative of sigmoid function times gradient of corresponding output
				grad_input_ptr->grad[0][col] = sigmoid(current_elem) * (1 - sigmoid(current_elem)) * forward_record.output_matrix.grad[0][col];
			}
		}


	private:

		// sigmoid function
		T sigmoid(T x)
		{
			// 1 / (1 + e^-x)
			return 1 / (1 + std::exp(-x));
		}


		// record of forward pass
		ForwardRecord<T, cols, cols> forward_record;
	};
}