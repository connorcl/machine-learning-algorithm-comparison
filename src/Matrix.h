#pragma once

#include <array>
#include <iostream>


namespace MLComparison
{
	// forward declaration of GradMatrix class template
	template<typename T, size_t n_rows, size_t n_cols>
	class GradMatrix;


	// class template for a matrix of a given type and dimensions
	template<typename T, size_t n_rows, size_t n_cols>
	class Matrix
	{
	public:

		// default constructor which initializes the data with zeroes
		Matrix() : data({ {{ 0 }} })
		{
		}


		// copy constructor which copies the data from the given matrix
		Matrix(const Matrix<T, n_rows, n_cols>& rhs) : data(rhs.data)
		{
		}


		// copy constructor which copies the data from the given gradient-enabled matrix
		Matrix(const GradMatrix<T, n_rows, n_cols>& rhs) : data(rhs.data)
		{
		}

		
		// virtual destructor
		virtual ~Matrix()
		{
		}


		// assignment operator which copies the data from the given matrix
		Matrix<T, n_rows, n_cols>& operator=(const Matrix<T, n_rows, n_cols>& rhs)
		{
			data = rhs.data;
			return *this;
		}


		// assignment operator which copies thr data from the given gradient-enabled matrix
		Matrix<T, n_rows, n_cols>& operator=(const GradMatrix<T, n_rows, n_cols>& rhs)
		{
			data = rhs.data;
			return *this;
		}


		// element access operator
		std::array<T, n_cols>& operator[](int i)
		{
			return data[i];
		}


		// const element access operator
		const std::array<T, n_cols>& operator[](int i) const
		{
			return data[i];
		}


		// element access method
		std::array<T, n_cols>& at(int i)
		{
			return data[i];
		}


		// const element access method
		const std::array<T, n_cols>& at(int i) const
		{
			return data[i];
		}


		// method template which calculates and returns the dot product of
		// the matrix and a given (possibly gradient-enabled) matrix
		template<template<typename, size_t, size_t> class MatrixType, size_t right_cols>
		Matrix<T, n_rows, right_cols> dot(const MatrixType<T, n_cols, right_cols>& rhs) const
		{
			// create a result matrix
			Matrix<T, n_rows, right_cols> result;

			// for each row of the result
			for (int row = 0; row < n_rows; row++)
			{
				// for each column of the result
				for (int col = 0; col < right_cols; col++)
				{
					// item is sum of elementwise product
					// of corresponding row of current matrix 
					// and corresponding column of matrix given as argument
					for (int k = 0; k < n_cols; k++)
					{
						result[row][col] += data[row][k] * rhs[k][col];
					}
				}
			}

			// return the result
			return result;
		}

		
		// method template which calculates and returns the dot product of the transpose
		// of the current matrix and a given (possibly gradient-enabled) matrix
		template<template<typename, size_t, size_t> class MatrixType, size_t right_cols>
		Matrix<T, n_cols, right_cols> t_dot(const MatrixType<T, n_rows, right_cols>& rhs) const
		{
			// create a result matrix
			Matrix<T, n_cols, right_cols> result;

			// for each row of the result
			for (int row = 0; row < n_cols; row++)
			{
				// for each column of the result
				for (int col = 0; col < right_cols; col++)
				{
					// item is sum of elementwise product
					// of corresponding row of transpose of current matrix
					// and corresponding column of matrix given as argument
					for (int k = 0; k < n_rows; k++)
					{
						result[row][col] += data[k][row] * rhs[k][col];
					}
				}
			}

			// return the result
			return result;
		}


		// method template which calculates and returns the dot product of the current
		// matrix and the transpose of a given (possibly gradient-enabled) matrix
		template<template<typename, size_t, size_t> class MatrixType, size_t right_rows>
		Matrix<T, n_rows, right_rows> dot_t(const MatrixType<T, right_rows, n_cols>& rhs) const
		{
			// create a result matrix
			Matrix<T, n_rows, right_rows> result;

			// for each row of the result
			for (int row = 0; row < n_rows; row++)
			{
				// for each column of the result
				for (int col = 0; col < right_rows; col++)
				{
					// item is sum of elementwise product
					// of corresponding row of current matrix and corresponding
					// column of transpose of matrix given as argument
					for (int k = 0; k < n_cols; k++)
					{
						result[row][col] += data[row][k] * rhs[col][k];
					}
				}
			}

			// return the result
			return result;
		}


		// method template which adds the elements of the given 
		// (possibly gradient-enabled) matrix to the current matrix
		template<template<typename, size_t, size_t> class MatrixType>
		void add_inplace(const MatrixType<T, n_rows, n_cols>& rhs)
		{
			// for each row of the current matrix
			for (int row = 0; row < n_rows; row++)
			{
				// for each column of the current matrix
				for (int col = 0; col < n_cols; col++)
				{
					// add the corresponding element of the
					// given matrix to that of the current matrix
					data[row][col] += rhs[row][col];
				}
			}
		}


		// method template which creates and returns a new matrix whose elements are the sum
		// of those in the current matrix and the given (possibly gradient-enabled) matrix
		template<template<typename, size_t, size_t> class MatrixType>
		Matrix<T, n_rows, n_cols> add(const MatrixType<T, n_rows, n_cols>& rhs) const
		{
			// copy the current matrix
			Matrix<T, n_rows, n_cols> result = *this;
			// add the given matrix to this copy
			result.add_inplace(rhs);
			// return the result
			return result;
		}


		// returns the transpose of the current matrix
		Matrix<T, n_cols, n_rows> t() const
		{
			// create a matrix to store the result
			Matrix<T, n_cols, n_rows> result;

			// for each row of the current matrix
			for (int row = 0; row < n_rows; row++)
			{
				// for each column of the current matrix
				for (int col = 0; col < n_cols; col++)
				{
					// set the diaglonally reflected element 
					// of the result matrix to the corresponding item
					// of the current matrix
					result[col][row] = data[row][col];
				}
			}

			// return the transposed matrix
			return result;
		}


		// prints the matrix
		void print() const
		{
			for (auto& row : data)
			{
				for (auto i : row)
				{
					std::cout << i << " ";
				}
				std::cout << std::endl;
			}
		}


		// two-dimensional array holding the data
		std::array<std::array<T, n_cols>, n_rows> data;
	};
}