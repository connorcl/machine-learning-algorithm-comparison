#pragma once

#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <iostream>


namespace MLComparison
{
	// class template for a tabular dataset suitable for a decision tree model
	template<typename T, int x_vars, bool includes_y = true>
	class DecisionTreeDataset
	{
	public:

		// default constructor which does not immediately load any data
		DecisionTreeDataset()
		{
		}


		// constructor which takes the name of a csv file from which data is to be loaded
		DecisionTreeDataset(const std::string& csv_file)
		{
			load_data(csv_file);
		}


		// element access operator which returns row i of the underlying data table
		auto& operator[](int i)
		{
			return data_table[i];
		}


		// const element access operator
		const auto& operator[](int i) const
		{
			return data_table[i];
		}


		// element access method
		auto& at(int i)
		{
			return data_table[i];
		}


		// const element access method
		const auto& at(int i) const
		{
			return data_table[i];
		}


		// returns an iterator pointing to the first element (row) of the data table
		auto begin()
		{
			return data_table.begin();
		}


		// returns an iterator pointing to the past-the-end element (row) of the data table
		auto end()
		{
			return data_table.end();
		}


		// returns an iterator pointing to the first element of the vector of row indices
		auto indices_begin()
		{
			return row_indices.begin();
		}


		// returns an iterator pointing to the past-the-end element of the vector of row indices
		auto indices_end()
		{
			return row_indices.end();
		}


		// returns an end iterator based on a number of rows to use
		auto indices_end(unsigned int rows_to_use)
		{
			// if rows to use is non-negative and less than number of rows in dataset
			if (rows_to_use < row_indices.size() && rows_to_use >= 0)
			{
				// create and return iterator pointing to desired element
				auto end_iterator = row_indices.begin();
				std::advance(end_iterator, rows_to_use);
				return end_iterator;
			}
			// otherwise, return end iterator of row indices vector
			else
			{
				return row_indices.end();
			}
		}


		// returns the number of rows in the dataset
		auto size()
		{
			return row_indices.size();
		}


		// returns the number of rows in the dataset
		auto get_n_rows()
		{
			return row_indices.size();
		}


		// returns the number of columns in the dataset
		auto get_n_cols()
		{
			return n_cols;
		}


		// returns the number of independent variables in the dataset
		auto get_n_x_vars()
		{
			return n_x_vars;
		}


		// loads data from a csv file
		void load_data(const std::string& csv_file)
		{
			// clear data table and row indices
			data_table.clear();
			row_indices.clear();
			// open csv file
			std::ifstream infile(csv_file);
			// create string and stream for storing the current line
			std::string line;
			std::istringstream line_stream;
			// create string to store the current item in the line/row
			std::string item;
			// for each row in the file, get the current line
			for (int row = 0; std::getline(infile, line); row++)
			{
				// add current row index to vector of row indices
				row_indices.emplace_back(row);
				// construct a new array in the vector of rows for the current row
				data_table.emplace_back();
				// set up line stream
				line_stream = std::istringstream(line);
				// for each field in the row
				for (int col = 0; col < n_cols; col++)
				{
					// get item from row
					std::getline(line_stream, item, ',');
					// convert string to number and store in data table
					data_table[row][col] = std::stod(item);
				}
			}
		}


		// outputs all the rows in the dataset
		void print()
		{
			for (auto& row : data_table)
			{
				for (auto i : row)
				{
					std::cout << i << " ";
				}
				std::cout << std::endl;
			}
		}


	private:

		// number of dependent variables in the dataset
		static const int n_x_vars = x_vars;
		// total number of columns in the dataset including the independent variable if present
		static const int n_cols = x_vars + (includes_y ? 1 : 0);

		// vector of arrays where each array corresponds to a data sample/row
		std::vector<std::array<T, n_cols>> data_table = {};
		// vector of row indices determining the order in which rows are accessed,
		// which is recursively partitioned into groups as a decision tree is trained
		std::vector<int> row_indices = {};
	};
}
