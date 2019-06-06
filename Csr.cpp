#include <iostream>
#include <sstream>
#include <string>
#include <math.h>

#include "Csr.h"

Csr::Csr ()
        : rows_ (0), columns_ (0), nnz_ (0), row_ptr_(0), col_ind_(0), val_(0) { }

Csr::Csr (int rows, int columns, int nnz)
        : rows_ (0), columns_ (0), nnz_ (0), row_ptr_(0), col_ind_(0), val_(0)
{
    setRows(rows);
    setColumns(columns);
    setNnz(nnz);
    createArrays();
}

Csr::~Csr (){
    delete [] row_ptr_;
    delete [] col_ind_;
    delete [] val_;
}

void Csr::createArrays (){
    row_ptr_ = new int [getRows()+1]();
    col_ind_ = new int [getNnz()]();
    val_ = new double [getNnz()]();
}

double Csr::getElement (int i, int j) const{
    for (int p = getRow_ptr(i); p < getRow_ptr(i+1); p ++){
        if (getCol_ind(p)-1 == j){
			return (getVal(p));
		} //end if
    } //end for
    return (0);
}

void Csr::transposeCsr (){
    int nrows2, ncols2, nnz2;
    nrows2 = getColumns();
    ncols2 = getRows();
    nnz2 = getNnz();

    int * row_ptr2 = new int [nrows2 + 1]();
    int * row_counts = new int [nrows2 + 1]();
    int * col_ind2 = new int [nnz2]();
    double * val2 = new double [nnz2]();

    /* first run: count the number of non-zero values in each column */
    for (int i = 0; i < getRows(); i ++){
        for (int j = getRow_ptr(i); j < getRow_ptr(i+1); j ++){
            int nnzIndex = getCol_ind(j);
            row_ptr2[nnzIndex] ++;
        }
    }

    /* construct the final row_ptr2 here */
    /* nnz values in each column (calculated in first run) become the nnz values in each row */
    int prev;
    for (int i = 0; i < nrows2+1; i ++){
        if (i == 0) {
            prev = 0;
        } else {
            prev = row_ptr2[i-1];
        } //end if
        row_ptr2[i] = row_ptr2[i] + prev;
    }

    /* second run: fill out the new matrix */
    for (int i = 0; i < getRows(); i ++){
        for (int j = getRow_ptr(i); j < getRow_ptr(i+1); j ++){
			int nnzIndex = getCol_ind(j);     //find "row" to place element in   
			         
            //transposedIndex calculates the correct index to transpose column index and values to
			int transposedIndex = ((row_ptr2[nnzIndex-1] + row_counts[nnzIndex]));	// will give a unique number
            col_ind2[transposedIndex] = i+1;	//convert to MatLab style
            val2[transposedIndex] = getVal(j);  
            row_counts[nnzIndex] ++;  //row_counts keeps track of how many nnz are in that row
        }
    }

    // Set new matrix
    setRows(nrows2);
    setColumns(ncols2);
    setNnz(nnz2);
    delete [] row_ptr_;
    delete [] col_ind_;
    delete [] val_;
    delete [] row_counts;   // not a data member
    row_ptr_ = row_ptr2;
    col_ind_ = col_ind2;
    val_ = val2;
}

double Csr::rowCosine (int i, int j){  
    return (modifiedRowCosine(i,j,0,0));
}

double Csr::modifiedRowCosine (int i, int j, int X, int Y){
    int nrowi = getRow_ptr(i+1) - getRow_ptr(i);
    int nrowj = getRow_ptr(j+1) - getRow_ptr(j);
    int ni = 0, nj = 0;
    double cosine = 0, lengthi = 0, lengthj = 0;
    
    while (ni < nrowi && nj < nrowj){
        int ci = (getRow_ptr(i) + ni);
        int cj = (getRow_ptr(j) + nj);
    
        if (getCol_ind(ci) == getCol_ind(cj)){
            cosine  += ( getVal(ci) - X) * ( getVal(cj) - Y);
            lengthi += ( getVal(ci) - X) * ( getVal(ci) - Y);
            lengthj += ( getVal(cj) - Y) * ( getVal(cj) - Y);
            ni ++;
            nj ++;
        }
        else if (getCol_ind(ci) > getCol_ind(cj)){
            lengthj += ( getVal(cj) - Y) * ( getVal(cj) - Y);
            nj ++;
        }
        else {
            lengthi += ( getVal(ci) - X) * ( getVal(ci) - X);
            ni ++;
        }
    }

	while (ni < nrowi){
		int ci = getRow_ptr(i) + ni;
		lengthi += ( getVal(ci) - X) * ( getVal(ci) - X);
		ni ++;
	}
	
	while (nj < nrowj){
		int cj = getRow_ptr(j) + nj;
		lengthj += ( getVal(cj) - Y) * ( getVal(cj) - Y);
		nj ++;
	}
    
    if (lengthi * lengthj){
        cosine /= sqrt(lengthi * lengthj);
    } else {
        cosine = 0;
    }
    
    return (cosine);
}

Csr * Csr::returnSimilarityMatrix () {
    /*This assumes current matrix is already transposed.
      Items should be in the rows. */

    /* Number of nnz equal to 1 */
    int newNnz; 
    newNnz = getRows();
    /* Check upper triangle of matrix, add 2 for each similarity not equal to 0 */
    for (size_t i = 0; i < getRows(); i ++){
        for (size_t j = i + 1; j < getRows(); j ++){
            double similarity = rowCosine(i,j);
            if (similarity != 0){
                newNnz += 2;
            }
        }
    }

    Csr * S = new Csr(getRows(), getRows(), newNnz);

    int nnzIndex = 0;
    for (size_t i = 0; i < getRows(); i ++){
        for (size_t j = 0; j < getRows(); j ++){
            double similarity = 1.0;
            if (i != j){
                similarity = rowCosine(i,j);
            }

            if (similarity != 0){
                S->setCol_ind(nnzIndex, j+1);
                S->setVal(nnzIndex, similarity);
                nnzIndex++;
            }
        }
        S->setRow_ptr(i+1, nnzIndex);
    }
    return (S);
}