#pragma once

#include <cmath>

#include "base_layers.h"


namespace MLComparison
{
	// class template for mean squared error (MSE) loss function layer
	template<typename T>
	class MSELoss : public LossLayer<T>
	{
	public:

		// forward pass which calculates and returns the mean squared error
		virtual T operator()(Matrix<T, 1, 1>* x, T target) override
		{
			// save pointer to input matrix
			this->input_matrix = x;
			this->target = target;
			// calculate and return loss
			return (x->at(0)[0] - target) * (x->at(0)[0] - target);
		}


		// backward pass which sets the gradients of the input matrix
		void backward() override
		{
			// gradient of input is 2 * error
			static_cast<GradMatrix<T, 1, 1>*>(this->input_matrix)->grad[0][0] = 2 * (this->input_matrix->at(0)[0] - this->target);
		}
	};
}