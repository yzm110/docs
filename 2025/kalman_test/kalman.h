#pragma once
#include <Eigen/Dense>
#include <iostream>

class kalman_filter{
public:
    kalman_filter(int state_num, int obsv_num);
    ~kalman_filter(){}

    virtual void setA() = 0;
    virtual void setH() = 0;
    virtual void setQ() = 0;
    virtual void setR() = 0;
    virtual void setP() = 0;

    void kalmanPredict();
    void kalmanCorrect();

    Eigen::MatrixXd get_state(){ return x_;}
    
    int stateNum_;
    int observationNum_;
    Eigen::MatrixXd x_; // state vector
    Eigen::MatrixXd z_; // observe vector
    Eigen::MatrixXd A_; // state transition matrix
    Eigen::MatrixXd H_; // obversation matrix
    Eigen::MatrixXd Q_; // state noise
    Eigen::MatrixXd R_; // observation noise
    Eigen::MatrixXd P_; // covariance matrix
};