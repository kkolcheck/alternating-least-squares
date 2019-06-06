#ifndef _CSR_H_
#define _CSR_H_

#include <iostream>
#include <sstream>
#include <string>

class Csr {
public:
    Csr ();
    Csr (int rows, int columns, int nnz);
    ~Csr ();

    void setRows (int rows);
    int getRows () const;
    void setColumns (int columns);
    int getColumns () const;
    void setNnz (int nnz);
    int getNnz () const;
    void setRow_ptr (int i, int value);
    int getRow_ptr (int i) const;
    void setCol_ind (int i, int value);
    int getCol_ind (int i) const;
    void setVal (int i, double value);
    double getVal (int i) const;

    void createArrays ();
    double getElement (int i, int j) const;
    void transposeCsr ();
    double rowCosine (int i, int j);
    double modifiedRowCosine (int i, int j, int X, int Y);
    Csr * returnSimilarityMatrix ();

private:
    int rows_;
    int columns_;
    int nnz_;
    int * row_ptr_;
    int * col_ind_;
    double * val_;
};

#include "Csr.inl"

#endif   // !defined _CSR_H