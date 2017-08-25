//
//  main.cu
//  cusp
//
//  Created by Elif Erbil on 10/07/2017.
//

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <iostream>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <chrono>

using namespace std;
using namespace chrono;

enum MatrixFormat { COO, CSR, DIA, ELL, HYB, FEATURE };

int main(int argc, char *argv[])
{
    time_point<system_clock> start, end;
    if (argc != 3) {
        cout << "I was expecting the matrix format and the matrix file path.\n";
    }
    
    int format = stoi(argv[1]);
    cout << "The matrix format must be an integer." << "\n";
    char *filename = argv[2];
   
    if(format == COO){
        cusp::coo_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cusp::coo_matrix<int, float, cusp::device_memory> A(A_host);
        // initialize input vector
        cusp::array1d<float, cusp::host_memory> x_host(A.num_cols);
        for (int i = 0; i < A.num_cols; i++)
            x_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> x(x_host);
        
        // allocate output vector
        cusp::array1d<float, cusp::host_memory> y_host(A.num_rows);
        for (int i = 0; i < A.num_rows; i++)
            y_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> y(y_host);
        
        // Warmup
        for(int i=0; i<5; i++){
            cusp::multiply(A, x, y);
        }
        
        int nz = A.num_entries;
        unsigned int ITERS;
        if (nz < 5000) {
            ITERS = 500000;
        } else if (nz < 10000) {
            ITERS = 200000;
        } else if (nz < 50000) {
            ITERS = 100000;
        } else if (nz < 100000) {
            ITERS = 50000;
        } else if (nz < 200000) {
            ITERS = 10000;
        } else if (nz < 1000000) {
            ITERS = 5000;
        } else if (nz < 2000000) {
            ITERS = 1000;
        } else if (nz < 3000000) {
            ITERS = 500;
        } else {
            ITERS = 200;
        }
        
        // Device computation
        
        start = system_clock::now();
        for(int i=0; i<ITERS; i++){
            cusp::multiply(A, x, y);
        }
        end = system_clock::now();
        duration<double> elapsed_seconds = end-start;
        
        cout << "Selected Format: COO" << "\n";
        cout << "time: " << elapsed_seconds.count()<< "s\n";
        cout << "iterations: " << ITERS << "s\n";
        cout << "elapsed time: " << elapsed_seconds.count()/ITERS << "s\n";
        // print the result
       // cusp::print(y);

    } else if(format == CSR){
        cusp::csr_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cusp::csr_matrix<int, float, cusp::device_memory> A(A_host);
        // initialize input vector
        cusp::array1d<float, cusp::host_memory> x_host(A.num_cols);
        for (int i = 0; i < A.num_cols; i++)
            x_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> x(x_host);
        
        // allocate output vector
        cusp::array1d<float, cusp::host_memory> y_host(A.num_rows);
        for (int i = 0; i < A.num_rows; i++)
            y_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> y(y_host);
        
        // Warmup
        for(int i=0; i<5; i++){
            cusp::multiply(A, x, y);
        }
        
        int nz = A.num_entries;
        unsigned int ITERS;
        if (nz < 5000) {
            ITERS = 500000;
        } else if (nz < 10000) {
            ITERS = 200000;
        } else if (nz < 50000) {
            ITERS = 100000;
        } else if (nz < 100000) {
            ITERS = 50000;
        } else if (nz < 200000) {
            ITERS = 10000;
        } else if (nz < 1000000) {
            ITERS = 5000;
        } else if (nz < 2000000) {
            ITERS = 1000;
        } else if (nz < 3000000) {
            ITERS = 500;
        } else {
            ITERS = 200;
        }
        
        // Device computation
        start = system_clock::now();
        for(int i=0; i<ITERS; i++){
            cusp::multiply(A, x, y);
        }
        end = system_clock::now();
        duration<double> elapsed_seconds = end-start;
        
        cout << "elapsed time: " << elapsed_seconds.count()/ITERS << "s\n";
        
        // ref computation
        for (int i = 0; i < A_host.num_rows; i++) {
            float sum = 0;
            for (int j = A_host.row_offsets[i]; j < A_host.row_offsets[i+1]; j++) {
                sum += A_host.values[j] * x_host[A_host.column_indices[j]];
            }
            y_host[i] = sum;
        }
        //cusp::print(y);
    }
    else if(format == DIA){
        cusp::dia_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cusp::dia_matrix<int, float, cusp::device_memory> A(A_host);
        // initialize input vector
        cusp::array1d<float, cusp::host_memory> x_host(A.num_cols);
        for (int i = 0; i < A.num_cols; i++)
            x_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> x(x_host);
        
        // allocate output vector
        cusp::array1d<float, cusp::host_memory> y_host(A.num_rows);
        for (int i = 0; i < A.num_rows; i++)
            y_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> y(y_host);
        
        // Warmup
        for(int i=0; i<5; i++){
            cusp::multiply(A, x, y);
        }
        
        int nz = A.num_entries;
        unsigned int ITERS;
        if (nz < 5000) {
            ITERS = 500000;
        } else if (nz < 10000) {
            ITERS = 200000;
        } else if (nz < 50000) {
            ITERS = 100000;
        } else if (nz < 100000) {
            ITERS = 50000;
        } else if (nz < 200000) {
            ITERS = 10000;
        } else if (nz < 1000000) {
            ITERS = 5000;
        } else if (nz < 2000000) {
            ITERS = 1000;
        } else if (nz < 3000000) {
            ITERS = 500;
        } else {
            ITERS = 200;
        }
        
        // Device computation
        start = system_clock::now();
        for(int i=0; i<ITERS; i++){
            cusp::multiply(A, x, y);
        }
        end = system_clock::now();
        duration<double> elapsed_seconds = end-start;
        
        cout << "elapsed time: " << elapsed_seconds.count()/ITERS << "s\n";
        // print the result
        //cusp::print(y);

    } else if(format == ELL){
        cusp::ell_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cusp::ell_matrix<int, float, cusp::device_memory> A(A_host);
        // initialize input vector
        cusp::array1d<float, cusp::host_memory> x_host(A.num_cols);
        for (int i = 0; i < A.num_cols; i++)
            x_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> x(x_host);
        
        // allocate output vector
        cusp::array1d<float, cusp::host_memory> y_host(A.num_rows);
        for (int i = 0; i < A.num_rows; i++)
            y_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> y(y_host);
        
        // Warmup
        for(int i=0; i<5; i++){
            cusp::multiply(A, x, y);
        }
        
        int nz = A.num_entries;
        unsigned int ITERS;
        if (nz < 5000) {
            ITERS = 500000;
        } else if (nz < 10000) {
            ITERS = 200000;
        } else if (nz < 50000) {
            ITERS = 100000;
        } else if (nz < 100000) {
            ITERS = 50000;
        } else if (nz < 200000) {
            ITERS = 10000;
        } else if (nz < 1000000) {
            ITERS = 5000;
        } else if (nz < 2000000) {
            ITERS = 1000;
        } else if (nz < 3000000) {
            ITERS = 500;
        } else {
            ITERS = 200;
        }
        
        // Device computation
        start = system_clock::now();
        for(int i=0; i<ITERS; i++){
            cusp::multiply(A, x, y);
        }
        end = system_clock::now();
        duration<double> elapsed_seconds = end-start;
        
        cout << "elapsed time: " << elapsed_seconds.count()/ITERS << "s\n";
        
        // print the result
        //cusp::print(y);

    } else if(format == HYB){
        cusp::hyb_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cusp::hyb_matrix<int, float, cusp::device_memory> A(A_host);
        // initialize input vector
        cusp::array1d<float, cusp::host_memory> x_host(A.num_cols);
        for (int i = 0; i < A.num_cols; i++)
            x_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> x(x_host);
        
        // allocate output vector
        cusp::array1d<float, cusp::host_memory> y_host(A.num_rows);
        for (int i = 0; i < A.num_rows; i++)
            y_host[i] = i + 1;
        cusp::array1d<float, cusp::device_memory> y(y_host);
        
        // Warmup
        for(int i=0; i<5; i++){
            cusp::multiply(A, x, y);
        }
        
        int nz = A.num_entries;
        unsigned int ITERS;
        if (nz < 5000) {
            ITERS = 500000;
        } else if (nz < 10000) {
            ITERS = 200000;
        } else if (nz < 50000) {
            ITERS = 100000;
        } else if (nz < 100000) {
            ITERS = 50000;
        } else if (nz < 200000) {
            ITERS = 10000;
        } else if (nz < 1000000) {
            ITERS = 5000;
        } else if (nz < 2000000) {
            ITERS = 1000;
        } else if (nz < 3000000) {
            ITERS = 500;
        } else {
            ITERS = 200;
        }
        
        // Device computation
        start = system_clock::now();
        for(int i=0; i<ITERS; i++){
            cusp::multiply(A, x, y);
        }
        end = system_clock::now();
        duration<double> elapsed_seconds = end-start;
        
        cout << "elapsed time: " << elapsed_seconds.count()/ITERS << "s\n";
        
        // print the result
        //cusp::print(y);

    }
    else if (format == FEATURE){
        cusp::csr_matrix<int, float, cusp::host_memory> A_host;
        cusp::io::read_matrix_market_file(A_host, filename);
        cout << "Matrix Features" << "\n";
        double n = A_host.num_rows;
        cout << "n: " << n << "\n";
        double nnz = A_host.num_entries;
        cout << "nnz: " << nnz << "\n";
        //dis
        double dis = 0;
        for(int i=0; i<A_host.num_rows; i++){
            double num = 0;
            double nz = 0;
            for (int j = A_host.row_offsets[i]; j < A_host.row_offsets[i+1]-1; j++) {
                num += A_host.column_indices[j+1]-A_host.column_indices[j];
                nz++;
            }
            dis += num/(nz+1);
        }
        cout << "dis: " << dis/n << "\n";
        //mu
        cout << "mu: ";
        for(int i=0; i<A_host.num_rows; i++){
            int num = 0;
            for (int j = A_host.row_offsets[i]; j < A_host.row_offsets[i+1]; j++) {
                num++;
            }
            cout << num << " ";
        }
        cout << "\n";
        //sd
        cout << "sd: ";
        double mean = nnz/n;
        for(int i=0; i<A_host.num_rows; i++){
            int num = 0;
            for (int j = A_host.row_offsets[i]; j < A_host.row_offsets[i+1]; j++) {
                num++;
            }
            cout << pow(num-mean, 2)/(n-1) << " ";
        }
        cout << "\n";
        //n*max
        cout << "n*max: ";
        int max = 0;
        for(int i=0; i<A_host.num_rows; i++){
            int num = 0;
            for (int j = A_host.row_offsets[i]; j < A_host.row_offsets[i+1]; j++) {
                num++;
            }
            if(num>max){
                max = num;
            }
        }
        cout << max*n << "\n";
    }
    else {
        cout << "Matrix type not found!" << "\n";
    }
    
}
