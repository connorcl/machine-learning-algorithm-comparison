#pragma once

#include "base_layers.h"
#include "Linear.h"
#include "Relu.h"
#include "Sigmoid.h"


namespace MLComparison
{
	// class template for a simple artificial neural network suitable for binary classification
	template<typename T, size_t input_cols>
	class NeuralNet : public TrainableLayer<T, input_cols, 1>
	{
	public:

		// constructor which takes a learning rate
		NeuralNet(T learning_rate) : TrainableLayer<T, input_cols, 1>(learning_rate), linear_layer_1(learning_rate), linear_layer_2(learning_rate)
		{
			// initialize weights of each linear layer
			linear_layer_1.kaiming_he_init();
			linear_layer_2.kaiming_he_init();
		}


		// call operator which performs the forward pass
		virtual Matrix<T, 1, 1>* operator()(Matrix<T, 1, input_cols>* x) override
		{
			return layer_2_sigmoid_activation(linear_layer_2(layer_1_relu_activation(linear_layer_1(x))));
		}


		// backward pass
		virtual void backward() override
		{
			layer_2_sigmoid_activation.backward();
			linear_layer_2.backward();
			layer_1_relu_activation.backward();
			linear_layer_1.backward();
		}


		// updates the parameters of both linear layers
		virtual void update() override
		{
			linear_layer_1.update();
			linear_layer_2.update();
		}


		// sets the learning rate
		virtual void set_lr(T new_learning_rate) override
		{
			this->learning_rate = new_learning_rate;
			linear_layer_1.set_lr(new_learning_rate);
			linear_layer_2.set_lr(new_learning_rate);
		}


	private:

		// first linear layer with 8 units (neurons)
		Linear<T, input_cols, 8> linear_layer_1;
		// relu activation function for first linear layer
		Relu<T, 8> layer_1_relu_activation;
		// second linear layer with a single unit
		Linear<T, 8, 1> linear_layer_2;
		// sigmoid activation function for second linear layer
		Sigmoid<T, 1> layer_2_sigmoid_activation;
	};
}