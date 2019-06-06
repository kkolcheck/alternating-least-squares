/* Getters and Setters */
inline
void Csr::setRows (int rows){
    rows_ = rows;
}

inline
int Csr::getRows () const{
    return (rows_);
}

inline
void Csr::setColumns (int columns){
    columns_ = columns;
}

inline
int Csr::getColumns () const{
    return (columns_);
}

inline
void Csr::setNnz (int nnz){
    nnz_ = nnz;
}

inline
int Csr::getNnz () const{
    return (nnz_);
}

inline
void Csr::setRow_ptr (int i, int value){
    row_ptr_[i] = value;
}

inline
int Csr::getRow_ptr (int i) const{
    return (row_ptr_[i]);
}

inline
void Csr::setCol_ind (int i, int value){
    col_ind_[i] = value;
}

inline
int Csr::getCol_ind (int i) const{
    return (col_ind_[i]);
}

inline
void Csr::setVal (int i, double value){
    val_[i] = value;
}

inline
double Csr::getVal (int i) const{
    return (val_[i]);
}