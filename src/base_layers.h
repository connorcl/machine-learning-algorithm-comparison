#pragma once

#include "GradMatrix.h"


namespace MLComparison
{
	// abstract class template for a neural network layer
	template<typename T, size_t input_size, size_t output_size>
	class Layer
	{
	public:

		// pure virtual function which takes a pointer to an input matrix, 
		// calculates an output matrix and returns a pointer to it
		virtual Matrix<T, 1, output_size>* operator()(Matrix<T, 1, input_size>*) = 0;

		// pure virtual function to calculate and set the relevant gradients
		virtual void backward() = 0;
	};


	// abstract class template for a layer with trainable parameters
	template<typename T, size_t input_size, size_t output_size>
	class TrainableLayer : public Layer<T, input_size, output_size>
	{
	public:

		// default constructor which sets the learning rate to 0
		TrainableLayer()
		{
		}


		// constructor which sets the learning rate to the given value
		TrainableLayer(T layer_learning_rate) : learning_rate(layer_learning_rate)
		{
		}


		// destructor
		~TrainableLayer()
		{
		}


		// getter for learning rate
		T get_lr()
		{
			return learning_rate;
		}

		
		// setter for learning rate
		virtual void set_lr(T new_learning_rate)
		{
			learning_rate = new_learning_rate;
		}

		
		// pure virtual function to update the layer's parameters
		virtual void update() = 0;


	protected:

		// learning rate for the layer's parameters
		T learning_rate = 0;
	};


	// class template for a loss function layer
	template<typename T>
	class LossLayer
	{
	public:

		// pure virtual call operator function which takes a pointer
		// to the input as well as the target number
		virtual T operator()(Matrix<T, 1, 1>* x, T target) = 0;

		// pure virtual function for backward pass
		virtual void backward() = 0;


	protected:

		// pointer to input matrix
		Matrix<T, 1, 1>* input_matrix = nullptr;
		// target
		T target = 0;
	};


	// struct template for a record of the forward pass, made up of
	// a pointer to the input matrix and the matrix of outputs
	template<typename T, size_t input_size, size_t output_size>
	struct ForwardRecord
	{
		// pointer to the matrix of inputs
		Matrix<T, 1, input_size>* input_matrix = nullptr;
		// matrix of outputs
		GradMatrix<T, 1, output_size> output_matrix;
	};
}