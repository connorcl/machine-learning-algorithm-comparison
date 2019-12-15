#pragma once

#include "base_layers.h"


namespace MLComparison
{
	// class template for a ReLU activation function layer whose inputs are of the given dimensions
	template<typename T, int cols>
	class Relu : public Layer<T, cols, cols>
	{
	public:

		// default constructor
		Relu()
		{
		}

		
		// forward pass which replaces all negative values with 0
		virtual Matrix<T, 1, cols>* operator()(Matrix<T, 1, cols>* x) override
		{
			// set pointer to address of matrix given as input
			forward_record.input_matrix = x;
			// temporary variable to hold the current element
			T current_elem;
			// for each column of input/output matrices
			for (int col = 0; col < cols; col++)
			{
				// get element from input matrix
				current_elem = x->at(0)[col];
				// set element of output matrix to current item if positive, else 0
				forward_record.output_matrix[0][col] = current_elem > 0 ? current_elem - 0.5 : -0.5;
			}

			// return pointer to output matrix
			return &forward_record.output_matrix;
		}


		// backward pass which sets gradient of each input to gradient of output if the input is positive, else 0
		void backward() override
		{
			// GradMatrix pointer to input matrix
			auto* grad_input_ptr = static_cast<GradMatrix<T, 1, cols>*>(forward_record.input_matrix);
			// for each column of input.output matrices
			for (int col = 0; col < cols; col++)
			{
				// gradient of input is gradient of corresponding output if input is positive, otherwise 0
				grad_input_ptr->grad[0][col] = forward_record.input_matrix->at(0)[col] > 0 ? forward_record.output_matrix.grad[0][col] : 0;
			}
		}


	private:

		// record of forward pass
		ForwardRecord<T, cols, cols> forward_record;
	};
}