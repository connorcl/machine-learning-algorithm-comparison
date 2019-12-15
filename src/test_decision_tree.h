#pragma once

#include <string>
#include <fstream>

#include "DecisionTreeModel.h"


namespace MLComparison
{
	// function to record the training time of the decision tree algorithm using
	// different numbers of training samples and independent variables within these
	void test_decision_tree(const std::string& train_timings_csv);
}