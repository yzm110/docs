#include "kf_instance.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

int main(){
    std::fstream mea_x, mea_v;
    string mea_x_path = "./mea_x2.txt";
    string mea_v_path = "./mea_v2.txt";
    mea_x.open(mea_x_path.c_str());
    mea_v.open(mea_v_path.c_str());
    // cout << "mea_x_path.c_str(): "<<mea_x_path.c_str() << endl;
    if (!mea_x.is_open()) {
        cout << "fail to open mea_x file!!!\n" << endl;
        return 0;
    }   
    if (!mea_v.is_open()) {
        cout << "fail to open mea_v file!!!\n" << endl;
        return 0;
    }  
    kf_instance kf_(2,2);
    stringstream post_x, post_v;
    post_x<<"post_x=[";
    post_v<<"post_v=[";
    stringstream mea_x_out, mea_v_out;
    mea_x_out<<"mea_x=[";
    mea_v_out<<"mea_v=[";    
    int count = 0;
    while(true){
        std::cout << "\n" << "          ************************       " << std::endl;
        string x_dataLine;
        string v_dataline;
        std::getline(mea_x, x_dataLine);
        std::getline(mea_v, v_dataline);

        if (x_dataLine.empty() || v_dataline.empty()) {
            post_x<<"]";
            post_v<<"]";
            std::cout<<post_x.str()<<std::endl;
            std::cout<<post_v.str()<<std::endl;
            mea_x_out<<"]";
            mea_v_out<<"]";
            std::cout<<mea_x_out.str()<<std::endl;
            std::cout<<mea_v_out.str()<<std::endl;
            break;
        }
        if(count !=0){
            post_x<<" ,";
            post_v<<" ,";
            mea_x_out<<" ,";
            mea_v_out<<" ,";            
        }
        kf_.z_(0)= std::stod(x_dataLine);
        kf_.z_(1)= std::stod(v_dataline);
        std::cout<<"z_(0): "<< kf_.z_(0)<<std::endl;
        std::cout<<"z_(1): "<< kf_.z_(1)<<std::endl;
        kf_.kalmanPredict();
        kf_.kalmanCorrect(); 
        post_x<<kf_.x_(0); 
        post_v<<kf_.x_(1); 
        mea_x_out<<x_dataLine;
        mea_v_out<<v_dataline;
        count++;     
    }

    return 0;
}