#pragma once

#include <string>
#include <fstream>

#include "NeuralNetModel.h"


namespace MLComparison
{
	// function template which generates a neural network model which uses a given
	// number of the independent variables in the banknote authentication dataset
	template<typename T, size_t x_vars_to_use>
	NeuralNetModel<T, 4, x_vars_to_use> make_banknote_authentication_nn_model(T learning_rate)
	{
		return NeuralNetModel<T, 4, x_vars_to_use>("banknote_train.csv", "banknote_valid.csv", learning_rate);
	}


	// function template to test a neural network model on a given number
	// of the independent variables in the banknote authentication dataset
	template<typename T, size_t x_vars_to_use>
	void test_model_with_n_x_vars(std::ofstream& timings_file)
	{
		// for each possible row number preset, i.e. using 1 to 8 8ths of the data samples
		for (int eighths_rows_to_use = 1; eighths_rows_to_use <= 8; eighths_rows_to_use++)
		{
			// create a model
			auto model = make_banknote_authentication_nn_model<T, x_vars_to_use>(0.1);
			// train the model and record the time taken
			auto train_time = model.train(eighths_rows_to_use, 5);
			// write details to timings file
			timings_file << eighths_rows_to_use << "," << x_vars_to_use << "," << train_time;
			// record the model's validation time
			auto valid_time = model.validate(eighths_rows_to_use);
			// write details to timings file
			timings_file << "," << valid_time << "," << model.get_accuracy() << std::endl;
		}
	}


	// function template to time the training of a neural network as
	// the input dimensions change
	template<typename T>
	void test_neural_network(const std::string& timings_csv)
	{
		// open the given timings file
		std::ofstream train_timings_file(timings_csv, std::ios::trunc);
		// write header to timings file
		train_timings_file << "samples_proportion,x_vars_proportion,train_time,valid_time,accuracy" << std::endl;

		// take 100 measurements of each combination of numbers of rows and columns to use
		for (int i = 0; i < 100; i++)
		{
			test_model_with_n_x_vars<T, 1>(train_timings_file);
			test_model_with_n_x_vars<T, 2>(train_timings_file);
			test_model_with_n_x_vars<T, 3>(train_timings_file);
			test_model_with_n_x_vars<T, 4>(train_timings_file);
		}
		// close timings file
		train_timings_file.close();
	}
}