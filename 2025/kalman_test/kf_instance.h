#include "kalman.h"

class kf_instance:public kalman_filter{
public:
    kf_instance(int state_num, int obsv_num);

    void setA() override;
    void setH() override;
    void setQ() override;
    void setR() override;
    void setP() override;

    void init();
};