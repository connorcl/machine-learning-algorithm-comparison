#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iterator>

#include "Matrix.h"


namespace MLComparison
{
	// class template for a dataset suitable for a neural network model
	// i.e each row is composed of a Matrix of independent variables
	// and a dependent variable which is a plain number
	template<typename T, size_t x_variables, size_t x_variables_to_use = x_variables>
	class NeuralNetDataset
	{
	public:

		// default constructor which leaves the dataset unpopulated
		NeuralNetDataset()
		{
		}


		// constructor which takes the name of a csv file to load
		NeuralNetDataset(const std::string& csv_file)
		{
			load_data(csv_file);
		}


		Matrix<T, 1, x_variables>& operator[](int i)
		{
			return data_table[i];
		}


		const Matrix<T, 1, x_variables>& operator[](int i) const
		{
			return data_table[i];
		}


		Matrix<T, 1, x_variables>& at(int i)
		{
			return data_table[i];
		}


		const Matrix<T, 1, x_variables>& at(int i) const
		{
			return data_table[i];
		}


		auto begin()
		{
			return data_table.begin();
		}


		auto end()
		{
			return data_table.end();
		}


		auto end(int rows_to_use)
		{
			if (rows_to_use < data_table.size() && rows_to_use >= 0)
			{
				auto end_iterator = data_table.begin();
				std::advance(end_iterator, rows_to_use);
				return end_iterator;
			}
			else
			{
				return data_table.end();
			}
		}


		auto size()
		{
			return data_table.size();
		}


		void load_data(const std::string& csv_file)
		{
			// open csv file
			std::ifstream infile(csv_file);
			// create string and stream to store each line in turn
			std::string line;
			std::istringstream line_stream;
			// create variable for each field
			std::string item;
			// for each row
			for (int row = 0; std::getline(infile, line); row++)
			{
				// construct new row in data table
				data_table.emplace_back();
				// create stream for line
				line_stream = std::istringstream(line);
				// for each independent variable to use
				for (int col = 0; col < x_variables_to_use; col++)
				{
					// get field from row
					std::getline(line_stream, item, ',');
					// set corresponding element in x dataset
					data_table[row].first[0][col] = std::stod(item);
				}
				// skip past each unused independent variable
				for (int col = 0; col < x_variables - x_variables_to_use; col++)
				{
					std::getline(line_stream, item, ',');
				}
				// get dependent variable
				std::getline(line_stream, item, ',');
				data_table[row].second = std::stod(item);
			}
		}


		void print()
		{
			for (auto& row : data_table)
			{
				for (auto i : row.first[0])
				{
					std::cout << i << " ";
				}
				std::cout << row.second << std::endl;
			}
		}


	private:

		std::vector<std::pair<Matrix<T, 1, x_variables_to_use>, T>> data_table;
	};
}