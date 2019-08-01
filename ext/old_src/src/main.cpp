#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

// #include <opencv2/opencv.hpp>

#include "lstm.hpp"
#include "matrix.hpp"
#include "activation.hpp"
#include "kernel.hpp"

using namespace std;

std::vector<char> readFile(const char * fileName) {
    std::ifstream infile(fileName);
    std::vector<char> buffer;
    //get length of file
    infile.seekg(0, infile.end);
    size_t length = infile.tellg();
    infile.seekg(0, infile.beg);
    //read file
    if (length > 0) {
        buffer.resize(length);    
        infile.read(&buffer[0], length);
    }
    cout << "File length: " << length << endl;
    return buffer;
}

int construct_int(vector<char> buffer, int index) {
    return (buffer[index] << 24) | (buffer[index + 1] << 16) | (buffer[index + 2] << 8) | buffer[index + 3];
}

int main(int argc, char **argv) {

    vector<char> buffer = readFile("mnist/t10k-images-idx3-ubyte");
    cout << construct_int(buffer, 0) << endl;
    cout << construct_int(buffer, 4) << endl;
    cout << construct_int(buffer, 8) << endl;
    cout << construct_int(buffer, 12) << endl;

    // readFile("mnist/t10k-labels-idx1-ubyte");
    // readFile("mnist/train-images-idx3-ubyte");
    // readFile("mnist/train-labels-idx1-ubyte");



    srand (time(NULL));
    
    int neurons = 30;

    cai::Matrix a_neuron_matrix(1, neurons, true);
    cai::Matrix b_neuron_matrix(1, neurons, true);

    cai::Matrix weight_matrix(neurons, neurons, false);
    cai::Matrix offset_matrix(1, neurons, true);
    weight_matrix.fill_rand(-1, 1);
    offset_matrix.fill_rand(-1, 1);

    a_neuron_matrix.fill_rand(0, 1);

    cai::Matrix *in_neuron_matrix = &a_neuron_matrix;
    cai::Matrix *out_neuron_matrix = &b_neuron_matrix;

    int itts = 100;

    for (int i = 0; i < itts; i++) {
        // cout << "------" << endl;

        cai::matrix_multiply(in_neuron_matrix, &weight_matrix, out_neuron_matrix);
        cai::matrix_add(out_neuron_matrix, &offset_matrix, out_neuron_matrix);
        cai::matrix_rectify(out_neuron_matrix, cai::sigmoid, 1);

        // out_neuron_matrix->print();

        // out_neuron_matrix->print();

        // cout << "Diff: " << cai::matrix_diff(in_neuron_matrix, out_neuron_matrix) << endl;

        cai::Matrix *temp = in_neuron_matrix;
        in_neuron_matrix = out_neuron_matrix;
        out_neuron_matrix = temp;

    }

    call_global();

}  


