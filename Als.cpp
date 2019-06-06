#include <iostream>
#include <string>
#include <math.h>   // std::fabs
#include "eigen/Eigen/Dense"
using Eigen::MatrixXd;

#include "Als.h"


Als::Als () { }

Als::Als (Csr * R, Csr * T, int k, double beta, bool random)
{
    int m = R->getRows();
    int n = R->getColumns();
    if (random == true) {
        setU(k, m, k);
        setV(k, n, k);
    } else {
        setU(k, m);
        setV(k, n);   
    }
    setUt();
    setVt();

    factorize(R, T, beta, k);
}

Als::~Als () { }

void Als::factorize(Csr * R, Csr * T, double beta, int k){
    MatrixXd lambda_diag = MatrixXd::Identity(k,k) * beta;

    double tolerance = 1;
    double prev_norm = 0.0;
    double round_norm = distance(R, k, beta);
    int counter = 0;  

    // Alternate between U and V functions until answer is convergent
    
    while (fabs(prev_norm - round_norm) > tolerance){
        prev_norm = round_norm;

        updateU(R, lambda_diag, k);
        updateV(R, lambda_diag, k);

        double dist = distance(R, k, beta);
        round_norm = dist;

        counter++;
        std::cout << "Iteration: " << counter << ". Threshold: " 
        << (prev_norm - round_norm) << " > 1" << std::endl;
        //std::cout << "RMSE R: " << rmse(R) << std::endl;
        std::cout << "RMSE T: " << rmse(T) << std::endl;
    }
}

void Als::updateU(Csr * R, MatrixXd lambda_diag, int k) {
    int m = R->getRows();
    int n = R->getColumns();

    //solve for U
    for (int u = 0; u < m; u++){
        // For each column of U
        MatrixXd sum_Vs = MatrixXd::Zero(k,k);
        MatrixXd sum_Vr = MatrixXd::Zero(k,1);

        for (int i = 0; i < n; i++){
            if (R->getElement(u,i) != 0){
                MatrixXd Vi = V_.col(i);    // (k x 1)
                MatrixXd Vit = Vt_.row(i);  // (1 x k)
                MatrixXd Vs_mult = Vi * Vit; // (k x 1) * (1 x k) = (k x k)

                sum_Vs = sum_Vs + Vs_mult;   // (k x k) + (k x k)
                
                //get each item in row u of matrix R
                sum_Vr = sum_Vr + (Vi * R->getElement(u,i));    // (k x 1) + [(k x 1) * scalar]
            }
        }

        MatrixXd inner_sum = lambda_diag + sum_Vs;      // (k x k) + (k x k)
        MatrixXd result = inner_sum.inverse() * sum_Vr; // inverse(k x k) * (k x 1) = (k x 1)
        for (int i = 0; i < k; i++){     // for each item in result         
            U_(i,u) = result(i,0);  // (k x 1)
        }
        setUt();    //set new transpose for U
    }
}

void Als::updateV(Csr * R, MatrixXd lambda_diag, int k) {
    int m = R->getRows();
    int n = R->getColumns();

    // solve for V
    for (int i = 0; i < n; i++){
        // For each column in V
        MatrixXd sum_Us = MatrixXd::Zero(k,k);
        MatrixXd sum_Ur = MatrixXd::Zero(k,1);

        for (int u = 0; u < m; u++){
            if (R->getElement(u,i) != 0){
            // For each column in U 
            MatrixXd Uu = U_.col(u);       // (k x 1)    
            MatrixXd Uut = Ut_.row(u);     // (1 x k)
            MatrixXd Us_mult = Uu * Uut;    // (k x 1) * (1 x k) = (k x k)
            
            sum_Us = sum_Us + Us_mult;   // (k x k) + (k x k)
            
            sum_Ur = sum_Ur + (Uu * R->getElement(u,i));            // (k x 1) + [(k x 1) * scalar]
            }
        } //end for u
        MatrixXd inner_sum = lambda_diag + sum_Us;                  // (k x k) + (k + k)
        MatrixXd result = inner_sum.inverse() * sum_Ur; // inverse(k x k) * (k x 1) = (k x 1)
        for (int j = 0; j < k; j++){
            V_(j,i) = result(j,0);  //(k x 1)
        }
        setVt();    //set new transpose for V
    }
}


double Als::distance(Csr * R, int k, double beta){
    //Objective Function
    int m = R->getRows();
    int n = R->getColumns();
    double dist = 0.0;
    MatrixXd Ut_V = getUt() * getV();    //(m x k) * (k x n) = (m x n)
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){   
            if (R->getElement(i,j) != 0)
                dist += (R->getElement(i,j) - Ut_V(i,j)) * (R->getElement(i,j) - Ut_V(i,j));
        }
    }
    dist = sqrt(dist);
    dist = dist * dist;
    dist = dist * 0.5;

    double Unorm = U_.norm();
    Unorm = Unorm * Unorm;

    double Vnorm = V_.norm();
    Vnorm = Vnorm * Vnorm;

    dist = dist + (beta/2.0)*(Unorm + Vnorm);
    
    return (dist);
}

double Als::rmse(Csr * T){

    int m = T->getRows();
    int n = T->getColumns();
    int nnz = T->getNnz();

    MatrixXd Ut_V = getUt() * getV();  // (m x k) * (k x n) = (m x n)
    double error = 0.0;

    //Summation for RMSE
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            double Tij = T->getElement(i,j);
            if (Tij != 0)
                error += (Tij - Ut_V(i,j)) * (Tij - Ut_V(i,j));  
        }
    }

    error = error/nnz;
    error = sqrt(error);

    return (error);
}

double Als::mae(Csr * T){
    int m = T->getRows();
    int n = T->getColumns();
    int nnz = T->getNnz();
    double error = 0.0;

    MatrixXd Ut_V = getUt() * getV();  // (m x k) * (k x n) = (m x n)

    //Summation for MAE
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            double Tij = T->getElement(i,j);
            if (Tij != 0)
                error += fabs(Tij - Ut_V(i,j));  
        }
    }
    error = error / nnz;
    return (error);    
}