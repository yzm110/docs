#include "kf_instance.h"

kf_instance::kf_instance(int state_num, int obsv_num) : kalman_filter(state_num, obsv_num){
    
    A_.resize(state_num, state_num);
    P_.resize(state_num, state_num);
    H_.resize(obsv_num,state_num);
    Q_.resizeLike(P_);
    R_.resize(obsv_num, obsv_num);
    x_.resize(state_num,1);
    z_.resize(obsv_num,1);
    
    init();
    setA();
    setH();
    setQ();
    setR();
    setP();
}

void kf_instance::setA(){
    //匀加速模型，deltaT =1 s
    A_<<1,1,0,1;
}

void kf_instance::setH(){
    //
    H_<<1,0,0,1;
}

void kf_instance::setQ(){
    //
    Q_<<0.1,0,0,0.1; //和R相反， 调大 后验结果靠近测量； 调太小，发散
}

void kf_instance::setR(){
    //
    R_<<1,0,0,1;
    // R_<<0.1,0,0,0.1; //调太大会让后验结果发散，调小会让后验结果贴近测量值，调到逼近0，后验结果就是测量值了
}

void kf_instance::setP(){
    //
    P_<<1,0,0,1;
    // P_<<100,0,0,100;//P的初值影响很小，值比较大的情况，会让第一个周期的后验估计状态贴近测量值，影响前几个周期的后验估计
}

void kf_instance::init(){
    x_<<0,1;
}

