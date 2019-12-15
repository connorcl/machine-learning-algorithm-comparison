#pragma once

#include "Matrix.h"


namespace MLComparison
{
	// class template for a matrix with a given type and dimensions which also
	// stores the gradients of some function with respect to its elements
	template<typename T, size_t n_rows, size_t n_cols>
	class GradMatrix : public Matrix<T, n_rows, n_cols>
	{
	public:

		// default constructor which calls the parent Matrix constructor
		GradMatrix() : Matrix<T, n_rows, n_cols>()
		{
		}


		// destructor
		~GradMatrix()
		{
		}


		// copy constructor which copies the data from a matrix
		GradMatrix(const Matrix<T, n_rows, n_cols>& rhs) : Matrix<T, n_rows, n_cols>(rhs)
		{
		}


		// copy constructor which copies only the data from a gradient-enabled matrix
		GradMatrix(const GradMatrix<T, n_rows, n_cols>& rhs) : Matrix<T, n_rows, n_cols>(rhs)
		{
		}


		// assignment operator which copies the data from a matrix
		GradMatrix<T, n_rows, n_cols>& operator=(const Matrix<T, n_rows, n_cols>& rhs)
		{
			this->data = rhs.data;
			return *this;
		}


		// assignment operator which copies only the data from a gradient-enabled matrix
		GradMatrix<T, n_rows, n_cols>& operator=(const GradMatrix<T, n_rows, n_cols>& rhs)
		{
			this->data = rhs.data;
			return *this;
		}


		// update the elements based on their gradients and the 
		// learning rate given, i.e. perform one gradient descent step
		void SGDStep(double learning_rate)
		{
			// for each row of the matrix
			for (int row = 0; row < n_rows; row++)
			{
				// for each column of the matrix
				for (int col = 0; col < n_cols; col++)
				{
					// multiply the relevant gradient by the learning rate
					// and subtract the result from the parameter
					this->data[row][col] -= learning_rate * grad[row][col];
				}
			}
		}

		
		// matrix which stores the gradients of some function
		// with respect to the elements of the data matrix
		Matrix<T, n_rows, n_cols> grad;
	};
}