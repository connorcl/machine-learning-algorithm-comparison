#pragma once

#include <chrono>
#include <string>
#include <cmath>
#include <memory>
#include <iterator>

#include "DecisionTreeDataset.h"
#include "DecisionTreeNode.h"
#include "calculate_rows_to_use.h"


namespace MLComparison
{
	// class template for a decision tree prediction model suitable for binary classification
	template<typename T, size_t dataset_x_vars>
	class DecisionTreeModel
	{
	public:

		// default constructor
		DecisionTreeModel()
		{
		}
	

		// constructor which takes the names of csv files for the training and validation sets
		DecisionTreeModel(const std::string& train_csv, const std::string& valid_csv) :
			validation_set(valid_csv)
		{
			// construct a new training dataset
			training_set_ptr.reset(new DecisionTreeDataset<T, dataset_x_vars>);
			// load training data
			training_set_ptr->load_data(train_csv);
		}


		// get accuracy
		T get_accuracy()
		{
			return validation_accuracy;
		}


		// loads a csv file as the training set
		void load_training_set_file(const std::string& csv_file)
		{
			training_set_ptr->load_data(csv_file);
		}


		// loads a csv file as the validation set
		void load_validation_set_file(const std::string& csv_file)
		{
			validation_set.load_data(csv_file);
		}


		// trains the model using a certain proportion of the training samples
		// and a given number of fields within these samples
		long long train(uint8_t eighths_rows_to_use, size_t x_vars_to_use)
		{
			// number of rows to use
			size_t rows_to_use = calculate_rows_to_use(8, eighths_rows_to_use, training_set_ptr->size());

			// get iterator pointing to beginning of training set row indices
			auto training_set_begin = training_set_ptr->indices_begin();
			// get iterator pointing to one after the last training sample to use
			auto training_set_end = training_set_ptr->indices_end(rows_to_use);

			// create the root node of the decision tree
			root_node_ptr.reset(new DecisionTreeNode<T, dataset_x_vars>(0, training_set_begin, training_set_end, training_set_ptr, x_vars_to_use));

			// get start time
			the_clock::time_point start = the_clock::now();

			// train the root node, which recursively creates and trains child nodes
			root_node_ptr->train();

			// get end time
			the_clock::time_point end = the_clock::now();

			// return number of microseconds taken
			return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		}


		// determine the model's accuracy using the validation set
		long long validate(uint8_t eighths_rows_to_use)
		{
			// number of rows to use
			size_t rows_to_use = calculate_rows_to_use(8, eighths_rows_to_use, validation_set.size());

			// total correct predictions
			T total_correct = 0;

			// get start time
			the_clock::time_point start = the_clock::now();

			// get end iterator
			auto end_iterator = validation_set.begin();
			std::advance(end_iterator, rows_to_use);
			// for each sample in the validation set
			for (auto it = validation_set.begin(); it < end_iterator; ++it)
			{
				// get sample
				auto& sample = *it;
				// get model's prediction
				auto prediction = root_node_ptr->predict(sample);
				// get target value
				auto target = sample[dataset_x_vars];
				// add whether prediction is correct to total correct predictions
				total_correct += (prediction == target);
			}

			// get end time
			the_clock::time_point end = the_clock::now();

			// calculate and record validation accuracy
			validation_accuracy = total_correct / validation_set.size();

			// total correct predictions
			total_correct = 0;

			// for each sample in the validation set
			for (auto& sample : validation_set)
			{
				// get model's prediction
				auto prediction = root_node_ptr->predict(sample);
				// get target value
				auto target = sample[dataset_x_vars];
				// add whether prediction is correct to total correct predictions
				total_correct += (prediction == target);
			}

			// calculate and record validation accuracy
			validation_accuracy = total_correct / validation_set.size();

			// return number of milliseconds taken
			return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		}

	
	private:

		// accuracy of the model on the validation set
		T validation_accuracy = 0;

		// shared pointer to the training set
		std::shared_ptr<DecisionTreeDataset<T, dataset_x_vars>> training_set_ptr = nullptr;
		// validation set
		DecisionTreeDataset<T, dataset_x_vars> validation_set;

		// unique pointer to the root node of the decision tree
		std::unique_ptr<DecisionTreeNode<T, dataset_x_vars>> root_node_ptr = nullptr;

		// alias for chrono::steady_clock used for performance measurement
		using the_clock = std::chrono::steady_clock;
	};
}