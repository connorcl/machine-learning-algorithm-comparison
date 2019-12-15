#include "calculate_rows_to_use.h"


namespace MLComparison
{
	// function to calculate a number of samples to use based on a number 
	// of divisions, a number of these to use, and the total number of rows
	size_t calculate_rows_to_use(int total_divisions, int divisions_to_use, size_t total_rows)
	{
		// if number of divisions is invalid (or equal to total)
		if (divisions_to_use >= total_divisions || divisions_to_use <= 0)
		{
			// use all training samples
			return total_rows;
		}
		// otherwise
		else
		{
			// calculate and return number of rows to use
			return static_cast<size_t>(std::round((static_cast<double>(divisions_to_use) / total_divisions)* total_rows));
		}
	}
}