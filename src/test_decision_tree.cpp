#include "test_decision_tree.h"


namespace MLComparison
{
	// function to record the training time of the decision tree algorithm using
	// different numbers of training samples and independent variables within these
	void test_decision_tree(const std::string& train_timings_csv)
	{
		// open timings file
		std::ofstream timings_file(train_timings_csv, std::ios::trunc);
		// write file header
		timings_file << "samples_proportion,x_vars_proportion,train_time,valid_time,accuracy" << std::endl;
		// take 30 measurements of each combination of numbers of rows and columns to use
		for (int i = 0; i < 100; i++)
		{
			// for each number of independed variables to use from 1 to 4
			for (int x_vars_to_use = 1; x_vars_to_use <= 4; x_vars_to_use++)
			{
				// for each number of eighths of the trianing samples to use from 1 to 8
				for (int eighths_rows_to_use = 1; eighths_rows_to_use <= 8; eighths_rows_to_use++)
				{
					// create model
					DecisionTreeModel<double, 4> model("banknote_train.csv", "banknote_valid.csv");
					// record training time
					auto train_time = model.train(eighths_rows_to_use, x_vars_to_use);
					// write details to timings file
					timings_file << eighths_rows_to_use << "," << x_vars_to_use << "," << train_time;
					// record validation time
					auto valid_time = model.validate(eighths_rows_to_use);
					// write details to csv file
					timings_file << "," << valid_time << "," << model.get_accuracy() << std::endl;
				}
			}
		}
	}
}