#pragma once

#include <cmath>


namespace MLComparison
{
	// function to calculate a number of samples to use based on a number 
	// of divisions, a number of these to use, and the total number of rows
	size_t calculate_rows_to_use(int total_divisions, int divisions_to_use, size_t total_rows);
}