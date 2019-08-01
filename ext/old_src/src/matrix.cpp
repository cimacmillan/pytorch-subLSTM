#include "matrix.hpp"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <cmath>

cai::Matrix::Matrix(int rows, int columns, bool x_indexed) {
    this->rows = rows;
    this->columns = columns;
    this->values = new float[rows * columns];

    //How the array is arranged for optimal access patterns
    this->row_factor = x_indexed ? this->columns : 1;
    this->column_factor = x_indexed ? 1 : this->rows;

    for (int i = 0; i < rows * columns; i++) {
        this->values[i] = 0;
    }

    // std::cout << "Matrix of size: (" << rows << ", " << columns << ")" << std::endl;
}

void cai::Matrix::set(int row, int column, float value) {
    this->values[(column * this->column_factor) + (row * this->row_factor)] = value;
}

void cai::Matrix::set_all(float values[]) {
    for (int i = 0; i < this->rows * this->columns; i++){
        int row_index = i / this->columns;
        int column_index = i % this->rows;

        this->set(row_index, column_index, values[i]);
    }
}

float cai::Matrix::get(int row, int column) {
    return this->values[(column * this->column_factor) + (row * this->row_factor)];
}

void cai::Matrix::print() {
    const char * line_delim =  " ------";
    for (int x = 0; x < this->columns; x++) {
        std::cout << line_delim;
    }
    std::cout << std::endl;
    for (int y = 0; y < this->rows; y++) {
        std::cout << "| ";
        for (int x = 0; x < this->columns; x++) {
            std::cout << std::fixed << std::setprecision(2) << this->values[x + (y * this->columns)] << " | ";
        }
        std::cout << std::endl;

        for (int x = 0; x < this->columns; x++) {
            std::cout << line_delim;
        }
        std::cout << std::endl;
    }
}

void cai::Matrix::fill_rand(float range_a, float range_b) {
    for (int i = 0; i < this->rows * this->columns; i++) {
        float random_float = ((float)rand()/(float)(RAND_MAX));
        float adjusted = ((1.0 - random_float) * range_a) + (random_float * range_b);
        this->values[i] = ((float)rand()/(float)(RAND_MAX)) * 2.0f - 1.0f;
    }
} 

//C needs to have correct bounds as does A and B
//A = (Q, W)
//B = (E, R)
//C = (Q, R)
//W == E
void cai::matrix_multiply(Matrix *a, Matrix *b, Matrix *c) {
    int c_columns = b->columns;
    int c_rows = a->rows;
    for (int y = 0; y < c_rows; y++) {
        for (int x = 0; x < c_columns; x++) {
            float result = 0;
            for (int i = 0; i < a->columns; i++) {
                result = result + (a->get(y, i) * b->get(i, x));
            }
            c->set(y, x, result);
        }
    }
}

void cai::matrix_add(Matrix *a, Matrix *b, Matrix *c) {
    int c_columns = b->columns;
    int c_rows = a->rows;
    for (int y = 0; y < c_rows; y++) {
        for (int x = 0; x < c_columns; x++) {
            c->set(y, x, (a->get(y, x) + b->get(y, x)));
        }
    }
}

void cai::matrix_rectify(Matrix *a, float (*f)(float, float), float alpha) {
    int a_columns = a->columns;
    int a_rows = a->rows;
    for (int y = 0; y < a_rows; y++) {
        for (int x = 0; x < a_columns; x++) {
            a->set(y, x, f(a->get(y, x), alpha));
        }
    }
}

float cai::matrix_diff(Matrix *a, Matrix *b) {
    float sum = 0;
    for (int y = 0; y < a->rows; y++) {
        for (int x = 0; x < a->columns; x++) {
            sum += fabs(a->get(y, x) - b->get(y, x));
        }
    }
    return sum / (a->rows * a->columns);
}

