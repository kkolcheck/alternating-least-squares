#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h> // fout
#include <chrono>
#include <ctime>
#include <math.h>   // std::fabs
#include "eigen/Eigen/Dense"

#include "Csr.h"
#include "Als.h"

using Eigen::MatrixXd;

Csr * fileIn (char * input);
void predictRating (char * output, Csr * T, Als myAls);


/*  6 input - user should input in this order:
    train.txt, test.txt, k, beta, lr, output.txt
        k = latent diminsion for matrix factors
        beta = regularization parameter
        lr = learning rate 
*/
int main(int argc, char * argv[]){
    auto start = std::chrono::system_clock::now();   // get time now
    int k = std::strtod(argv[3], NULL);
    double beta = std::strtod(argv[4], NULL);
    double lr = std::strtod(argv[5], NULL);
    Csr * R;    //training user-item matrix
    Csr * T;    //testing user-item matrix
    Csr * P;    //rating prediction matrix
    R = fileIn(argv[1]); 
    T = fileIn(argv[2]);

    bool random = false;
    Als myAls(R, T, k, beta, random);
    predictRating (argv[6], T, myAls);
    std::cout << std::endl;
    std::cout << "Final RMSE: " << myAls.rmse(T) << std::endl;
    std::cout << "Final MAE: " << myAls.mae(T) << std::endl;

    delete R;
    delete T;

    auto end = std::chrono::system_clock::now();    // get time now
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);   
    std::cout << "Finished computation at " << std::ctime(&end_time) << ">> Elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
} // end main

Csr * fileIn (char * input){
    std::ifstream inFile;
    inFile.open(input);

    //Read and store first line of file
    std::string line;
    getline(inFile, line);

    //store line as streamstring
    std::stringstream ssline;
    ssline << line;    
    double token;

    //Parse matrix details
    ssline >> token;
    int rows = token;
    ssline >> token;
    int col = token;
    ssline >> token;
    int nnz = token;
    Csr * inMatrix = new Csr(rows, col, nnz);

    /* Import CSR into appropriate arrays */
    inMatrix->setRow_ptr (0,0);
    int nnzIndex = 0;
    for (int i = 0; i < inMatrix->getRows(); i ++){
        getline(inFile, line);
        ssline.str("");
        ssline.clear();
        ssline << line;
        while (ssline >> token){
            // int values in document are in pairs
            inMatrix->setCol_ind(nnzIndex, token);
            ssline >> token;
            inMatrix->setVal(nnzIndex, token);
            nnzIndex ++;
        } //end while
        inMatrix->setRow_ptr(i + 1, nnzIndex);     // Keep track of how many non zero entries in each row
    } //end for
    inFile.close();

    return inMatrix;
}

void predictRating (char * output, Csr * T, Als myAls){
    MatrixXd Ut_V = myAls.getUt() * myAls.getV();  // (m x k) * (k x n) = (m x n)
    
    int m = T->getRows();
    int n = T->getColumns();
    int nnz = T->getNnz();
    
    //output matrix
    std::ofstream outFile;  
    outFile.open(output);
    outFile << m << " " << n << " " << nnz << std::endl;

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (T->getElement(i,j) != 0){
                int num = round(fabs(Ut_V(i,j)));
                outFile << j + 1 << " " << num << " ";
            }
        }
        outFile << std::endl;  
    }
}