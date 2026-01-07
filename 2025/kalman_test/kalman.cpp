#include "kalman.h"

kalman_filter::kalman_filter(int state_num, int obsv_num){
    stateNum_ = state_num;
    observationNum_ = obsv_num;
    A_ = Eigen::MatrixXd::Zero(stateNum_,stateNum_);
    H_ = Eigen::MatrixXd::Zero(observationNum_,stateNum_);
    Q_ = Eigen::MatrixXd::Zero(stateNum_,stateNum_); 
    R_ = Eigen::MatrixXd::Zero(observationNum_,observationNum_);
    P_ = Eigen::MatrixXd::Identity(stateNum_,stateNum_);
}

void kalman_filter::kalmanPredict(){
    assert(A_.rows() == P_.rows());
    assert(Q_.rows() == P_.rows());
    std::cout<<"P_k-1: "<<P_<<std::endl;
    // propagate system covariance and error state
    x_ = A_*x_;
    P_ = A_*P_*A_.transpose() + Q_;
    std::cout<<"x_hat: "<<x_<<std::endl;
    std::cout<<"P_hat: "<<P_<<std::endl;
}

void kalman_filter::kalmanCorrect(){
    assert(H_.cols() == P_.rows());
    assert(z_.rows() == H_.rows());
    assert(z_.rows() == R_.rows());
    assert(z_.cols() == 1);
    //// Compute Kalman Gain
    auto temp  = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * temp.inverse();

    // 更新系统误差状态和协方差
    Eigen::MatrixXd I;
    I.resizeLike(P_);
    I.setIdentity();
    I = I - K * H_;
    // 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz;
    x_  = x_ + K * (z_ - H_ * x_);
    // P_ = I * Cov_ * I.transpose() + K * R * K.transpose();  //协方差矩阵更新的约瑟夫形式
    P_ = I*P_;  //简化形式
}