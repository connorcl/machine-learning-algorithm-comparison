#include <iostream>
#include <string>

#include "test_neural_network.h"
#include "test_decision_tree.h"


int main()
{
	// output filenames
	std::string deep_learning_output_file = "deep_learning_results.csv";
	std::string decision_tree_output_file = "decision_tree_results.csv";

	// test each algorithm and output timings to file
	std::cout << "Training and validating deep learning algorithm... (Writing results to " << deep_learning_output_file << ")" << std::endl;
	MLComparison::test_neural_network<float>(deep_learning_output_file);
	std::cout << "Training and validating decision tree algorithm... (Writing results to " << decision_tree_output_file << ")" << std::endl;
	MLComparison::test_decision_tree(decision_tree_output_file);

	return 0;
}