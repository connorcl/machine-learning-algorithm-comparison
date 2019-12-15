#pragma once

#include <chrono>
#include <string>
#include <cmath>

#include "NeuralNetDataset.h"
#include "NeuralNet.h"
#include "MSELoss.h"
#include "calculate_rows_to_use.h"


namespace MLComparison
{
	// class template for a neural network prediction model suitable for binary classification
	template<typename T, size_t dataset_x_vars, size_t model_x_vars = dataset_x_vars>
	class NeuralNetModel
	{	
	public:

		// default constructor
		NeuralNetModel()
		{
		}


		// constructor which takes filenames for the training and validation sets and the network's learning rate
		NeuralNetModel(const std::string& train_csv, const std::string& valid_csv, T learning_rate) :
			training_set(train_csv),
			validation_set(valid_csv),
			neural_net(learning_rate)
		{
		}


		// get accuracy
		T get_accuracy()
		{
			return validation_accuracy;
		}


		// sets the learning rate
		void set_learning_rate(T new_learning_rate)
		{
			neural_net.set_lr(new_learning_rate);
		}


		// loads a csv file as the training set
		void load_training_set_file(const std::string& csv_file)
		{
			training_set.load_data(csv_file);
		}


		// loads a csv file as the validation set
		void load_validation_set_file(const std::string& csv_file)
		{
			validation_set.load_data(csv_file);
		}

		
		// trains the neural network on the given subset of the
		// rows of the dataset and for a given number of epochs
		long long train(uint8_t eighths_rows_to_use, size_t n_epochs)
		{
			// number of rows to use, calculated from the given preset number
			size_t rows_to_use = calculate_rows_to_use(8, eighths_rows_to_use, training_set.size());

			// get start time
			the_clock::time_point start = the_clock::now();

			// for each epoch
			for (size_t epoch = 0; epoch < n_epochs; epoch++)
			{
				// for each training sample to use
				for (auto it = training_set.begin(); it < training_set.end(rows_to_use); ++it)
				{
					// get reference to current sample
					auto& row = *it;
					// perform forward pass
					loss(neural_net(&row.first), row.second);
					// perform backward pass
					loss.backward();
					neural_net.backward();
					// update the parameters
					neural_net.update();
				}
			}

			// get end time
			the_clock::time_point end = the_clock::now();

			// return number of microseconds taken
			return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		}


		// determine the model's accuracy using the validation set
		long long validate(uint8_t eighths_rows_to_use)
		{
			// number of rows to use, calculated from the given preset number
			size_t rows_to_use = calculate_rows_to_use(8, eighths_rows_to_use, validation_set.size());

			// get start time
			the_clock::time_point start = the_clock::now();

			// get end iterator
			auto end_iterator = validation_set.begin();
			std::advance(end_iterator, rows_to_use);
			// for each sample in the validation set
			for (auto it = validation_set.begin(); it < end_iterator; ++it)
			{
				// get sample
				auto& row = *it;
				// calculate the model's prediction
				neural_net(&row.first);
			}

			// get end time
			the_clock::time_point end = the_clock::now();

			// total loss over all samples
			T total_loss = 0;
			// total correct predictions
			T total_correct = 0;

			// for each sample in the validation set
			for (auto& row : validation_set)
			{
				// calculate the model's prediction
				auto prediction = neural_net(&row.first);
				// get the target value
				auto target = row.second;
				// calculate the MSE of the model's prediction
				total_loss += loss(prediction, target);
				// add whether the prediction was correct to the total of correct predictions
				total_correct += (std::round(prediction->at(0)[0]) == std::round(target));
			}

			// calculate and save the average loss and the accuracy
			validation_loss = total_loss / validation_set.size();
			validation_accuracy = total_correct / validation_set.size();

			// return the number of microseconds taken
			return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		}


	private:

		// validation loss and accuracy
		T validation_loss = 0;
		T validation_accuracy = 0;

		// training and validation sets
		NeuralNetDataset<T, dataset_x_vars, model_x_vars> training_set;
		NeuralNetDataset<T, dataset_x_vars, model_x_vars> validation_set;
		
		// neural network itself
		NeuralNet<T, model_x_vars> neural_net;
		
		// mean squared error loss function object
		MSELoss<T> loss;

		// alias for chrono::steady_clock used for performance measurement
		using the_clock = std::chrono::steady_clock;
	};
}
