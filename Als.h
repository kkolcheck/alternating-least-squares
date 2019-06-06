// Alternating Least Squares

#ifndef _ALS_H_
#define _ALS_H_

#include <iostream>
#include <string>
#include "eigen/Eigen/Dense"

using Eigen::MatrixXd;

#include "Csr.h"

class Als {
public:
    Als ();
    Als (Csr * R, Csr * T, int k, double beta, bool random);
    ~Als ();
    void setU(int row, int col);             //Random
    void setU(int row, int col, int k);    //Constant
    void setV(int row, int col);             //Random
    void setV(int row, int col, int k);    //Constant
    void setUt();
    void setVt();
    MatrixXd getU();
    MatrixXd getV();
    MatrixXd getUt();
    MatrixXd getVt();
    void factorize(Csr * R, Csr * T, double lr, int k);
    void updateU(Csr * R, MatrixXd lambda_diag, int k);
    void updateV(Csr * R, MatrixXd lambda_diag, int k);
    double distance(Csr * R, int k, double beta);
    double rmse(Csr * T);
    double mae(Csr * T);
protected:
    MatrixXd U_;
    MatrixXd V_;
    MatrixXd Ut_;
    MatrixXd Vt_;
};

#include "Als.inl"

#endif   // !defined _ALS_H