#pragma once

#include <vector>
#include <array>
#include <memory>
#include <iterator>
#include <iostream>
#include <algorithm>

#include "DecisionTreeDataset.h"


namespace MLComparison
{
	// class template for a binary classification decision tree node which deals with data samples of a certain type and size
	template<typename T, int dataset_x_vars>
	class DecisionTreeNode
	{
	public:

		// constructor which takes the depth of the current node, iterators defining the training group,
		// a pointer to the training dataset, and the number of training set indepenent variables to use
		DecisionTreeNode(int node_depth, std::vector<int>::iterator group_begin_it, std::vector<int>::iterator group_end_it,
			std::shared_ptr<DecisionTreeDataset<T, dataset_x_vars>> training_dataset, int dataset_x_vars_to_use = dataset_x_vars) :
			depth(node_depth), group_begin(group_begin_it), group_end(group_end_it), 
			training_set(training_dataset), x_vars_to_use(dataset_x_vars_to_use)
		{
			group_size = std::distance(group_begin, group_end);
		}


		// copy constructor, which does not copy child nodes or pointers to them
		DecisionTreeNode(const DecisionTreeNode<T, dataset_x_vars>& rhs) :
			split_var(rhs.split_var),
			split_val(rhs.split_val),
			class_prediction(rhs.class_prediction),
			depth(rhs.depth),
			left(nullptr),
			right(nullptr),
			training_set(rhs.training_set),
			x_vars_to_use(rhs.x_vars_to_use),
			group_begin(rhs.group_begin),
			group_end(rhs.group_end),
			group_size(rhs.group_size)
		{
		}


		// assignment operator, which does not copy child nodes or pointers to them
		DecisionTreeNode<T, dataset_x_vars>& operator=(const DecisionTreeNode<T, dataset_x_vars>& rhs)
		{
			split_var = rhs.split_var;
			split_val = rhs.split_var;
			class_prediction = rhs.class_prediction;
			depth = rhs.depth;
			left.reset(nullptr);
			right.reset(nullptr);
			training_set = rhs.training_set;
			x_vars_to_use = rhs.x_vars_to_use;
			group_begin = rhs.group_begin;
			group_end = rhs.group_end;
			group_size = rhs.group_size;
			return *this;
		}


		// trains the node on its group of training samples, determining the optimal split variable and value
		// and either setting its class prediction or recursively creating and training its child nodes
		void train()
		{
			// if depth is low enough and group size large enough
			if (depth < 6 && group_size > 10)
			{
				// determine best split variable and value
				get_best_split();
				// partition the node's group of training row indices and return an iterator pointing to the split point
				auto split_point = split_group();
				// become leaf node if best split is not to split
				if (split_point == group_begin || split_point == group_end)
				{
					become_leaf();
				}
				// otherwise, create and train child nodes
				else
				{
					left.reset(new DecisionTreeNode<T, dataset_x_vars>(depth + 1, group_begin, split_point, training_set, x_vars_to_use));
					right.reset(new DecisionTreeNode<T, dataset_x_vars>(depth + 1, split_point, group_end, training_set, x_vars_to_use));
					left->train();
					right->train();
				}
			}
			// otherwise, become leaf node
			else
			{
				become_leaf();
			}
		}


		// method template for making a prediction based on a sample
		template<int sample_length>
		int predict(const std::array<T, sample_length>& sample)
		{
			// if node is leaf node, return prediction
			if (class_prediction >= 0)
			{
				return class_prediction;
			}
			// if node is not leaf and sample's split variable is less than split value,
			// return left child node's prediction
			else if (sample[split_var] < split_val)
			{
				return left->predict(sample);
			}
			// otherwise, return right child node's prediction
			else
			{
				return right->predict(sample);
			}
		}


	private:

		// default constructor does not make sense for this class, so it is kept private and without definition
		DecisionTreeNode();


		// calculates the Gini index of a split point defined by a split variable and value
		double calculate_gini_index(int split_variable, T split_value)
		{
			// array of subgroup sizes
			std::array<double, 2> subgroup_sizes = {};
			// array of subgroup class value sums
			std::array<double, 2> subgroup_class_val_sums = {};
			
			// subgroup current row belongs in
			uint8_t row_subgroup = 0;
			// for each row in the group
			for (auto it = group_begin; it < group_end; ++it)
			{
				// get a reference to the current row
				const auto& row = training_set->at(*it);
				// determine which subgroup the row belongs in
				row_subgroup = row[split_variable] >= split_value;
				// increment the revelvant subgroup size and class value sum appropriately
				++subgroup_sizes[row_subgroup];
				subgroup_class_val_sums[row_subgroup] += row[dataset_x_vars];
			}

			// subgroup sizes as proportions
			std::array<double, 2> subgroup_size_proportions;
			subgroup_size_proportions[0] = subgroup_sizes[0] / group_size;
			subgroup_size_proportions[1] = 1 - subgroup_size_proportions[0];

			// proportion of each class in current group
			std::array<double, 2> class_proportions;
			// final gini index
			double gini_index = 0;

			// for each subgroup
			for (int subgroup = 0; subgroup <= 1; subgroup++)
			{
				// prevent division by zero if a group has 0 size
				if (subgroup_sizes[subgroup] > 0)
				{
					// proportion of positive class is sum of group class values / size of group
					class_proportions[1] = subgroup_class_val_sums[subgroup] / subgroup_sizes[subgroup];
					// proportion of negative class is 1 - proportion of positive class
					class_proportions[0] = 1 - class_proportions[1];
					// calculate the gini index for the group, weight it by subgroup size and add it to the final gini index
					gini_index += (1 - ((class_proportions[0] * class_proportions[0]) + (class_proportions[1] * class_proportions[1]))) * subgroup_size_proportions[subgroup];
				}
			}

			// return final gini index
			return gini_index;
		}


		// find the split point (variable and value) in this node's training group which has
		// the lowest gini index, and set the node's split variable and value accordingly
		void get_best_split()
		{
			// split value being tested
			T current_val = 0;
			// gini index of current split point
			double current_gini_index = 0;

			// best gini index found
			double best_gini_index = 0.5;
			// split value associated with best gini index
			double best_val = 0;
			// split variable associated with best gini index
			int best_var = 0;

			// whether perfect gini index has been found
			bool done = false;

			// for each row index of the current group
			for (auto it = group_begin; it < group_end && !done; ++it)
			{
				// get reference to current row
				const auto& row = training_set->at(*it);
				// for each field to use in the current row
				for (int col = 0; col < x_vars_to_use && !done; col++)
				{
					// get the field's value
					current_val = row[col];
					// calculate Gini index of split at current row & col
					current_gini_index = calculate_gini_index(col, current_val);
					// update best if necessary
					if (current_gini_index < best_gini_index)
					{
						best_gini_index = current_gini_index;
						best_val = current_val;
						best_var = col;
						// stop if perfect Gini score has been found
						if (best_gini_index == 0)
						{
							done = true;
						}
					}
				}
			}

			// set split variable and value fields once the best are found
			split_var = best_var;
			split_val = best_val;
		}


		// paritions the node's group of rows based on its split point and returns
		// an iterator pointing to the split point (first row index of the second group)
		auto split_group()
		{
			// sort group based on split variable and value and return iterator pointing to split point (first element of second group)
			return std::partition(group_begin, group_end,
				[this](int i) -> bool { 
					return training_set->at(i)[split_var] < split_val;
				}
			);
		}


		// set the leaf prediction value based on node's group of training samples
		void become_leaf()
		{
			// get sum of class values in group
			int sum = 0;
			for (auto it = group_begin; it < group_end; ++it)
			{
				sum += training_set->at(*it)[dataset_x_vars];
			}
			// set leaf class prediction based on whichever class is more prevalent
			class_prediction = (sum > group_size / 2) ? 1 : 0;
		}


		// split variable and value if the node is not a leaf
		int split_var = -1;
		T split_val = -1;
		// class prediction if the node is a leaf
		int class_prediction = -1;

		// depth of node in tree
		int depth;

		// pointers to left and right child nodes
		std::unique_ptr<DecisionTreeNode<T, dataset_x_vars>> left = nullptr;
		std::unique_ptr<DecisionTreeNode<T, dataset_x_vars>> right = nullptr;

		// pointer to the dataset on which to train
		std::shared_ptr<DecisionTreeDataset<T, dataset_x_vars>> training_set;

		// number of dependent variables in the dataset to use
		int x_vars_to_use;

		// iterators pointing to the elements in the training set's vector of row indices
		// which define the start and end of the group of samples on which to train
		std::vector<int>::iterator group_begin;
		std::vector<int>::iterator group_end;

		// number of rows in the node's training group
		int group_size = 0;
	};
}
