/* Getters and Setters */
inline
void Als::setU(int row, int col){
    U_.setRandom(row, col);
}

inline
void Als::setU(int row, int col, int k){
    U_ = MatrixXd::Zero(row,col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){   
            U_(i,j) = 1.0/k;
        }
    }
}

inline
void Als::setV(int row, int col){
    V_.setRandom(row, col);
}

inline
void Als::setV(int row, int col, int k){
    V_ = MatrixXd::Zero(row,col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){   
            V_(i,j) = 1.0/k;
        }
    }
}

inline
void Als::setUt(){
    Ut_ = U_.transpose();
}

inline
void Als::setVt(){
    Vt_ = V_.transpose();
}

inline
MatrixXd Als::getU(){
    return (U_);
}

inline
MatrixXd Als::getV(){
    return (V_);
}

inline
MatrixXd Als::getUt(){
    return (Ut_);
}

inline
MatrixXd Als::getVt(){
    return (Vt_);
}