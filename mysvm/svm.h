#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
class MYSVM
{
private:
/**
 * @brief selfdata
 * @param data data
 * @param label label
 * @param C slack variety
 * @param tol contain the error rate
 */
    Eigen::MatrixXd _data;
    Eigen::MatrixXd _label;
    float _C;
    float _tol;
/**
 * @brief model parameter
 * @param data_num num of data (num of row)
 * @param alphas using to calculate dual-problem (init to zeros)
 * @param b parameter to construct a hyperplane
 * @param error_box error matrix size(data_num,2), first is "whether it is valid" 
 *               second is the error of the Sample
 */
   uint _data_num;
   Eigen::MatrixXd _alphas;
   Eigen::MatrixXd _error_box;
   float _b;
/**
 * @brief kernel
 * 
 */
   Eigen::MatrixXd Kernel;
/**
 * @brief init to control smo
 * @param _afterInit a flag
 */
   bool _afterInit = false;
public:
/**
 * @brief Construct a new MYSVM object
 * @brief init the parameters
 * 
 */
    MYSVM(){};
    ~MYSVM(){};
/**
 * @brief init the data for kernel and other private variety
 * refered in loadfile()
 * @param datain input data
 * @param labels data's label
 * @param c      slack variety
 * @param toler contain the error rate
 * @brief 初始化svm分类器
 * @return int flag for tips
 */
   int _init_SVM(Eigen::MatrixXd datain , Eigen::MatrixXd labels , float c , float toler);
 /**
  * @brief load data
  * refered in main to get path for data
  * @param filepath data_path
  * @return int flag for tips
  * @brief 装填文件的数据
  */
   int load_file(string filepath);
 /**
  * @brief count the kernel
  * refered in loadfile()
  * @param data input_data
  * @param k    gauss var
  * @return Eigen::MatrixXd 
  * @brief 径向基函数将数据映射到多维超空间
  */
   Eigen::MatrixXd radial_bassic_func(Eigen::MatrixXd data , float k);
 /**
  * @brief smo
  * 
  * @param max_for max times for circle
  * @return int 
  * @brief smo算法，计算对偶问题的alpha
  */
   int smo(int max_for);
 /**
  * @brief using in smo
  * using in smo count the alpha
  * @param ord the oreder of the ord_th sample
  * @return int 
  */
   int innerL(int ord);
 /**
  * @brief calculate the error of the ord_th sample
  * using in innerL
  * @param ord the oreder of the ord_th sample
  * @return int
  * @brief 计算序号为order的误差
  */
   float calculate_error(int ord);
  /**
   * @brief select the max growth_aim_j
   * 
   */
   Eigen::Vector2f select_j(int i, float Ei);
  /**
   * @brief return the list of noneZero
   * 
   */
   Eigen::VectorXf valid_list(Eigen::MatrixXd error_box);
  /**
   * @brief select aa random j between 0 and _data_num
   * 
   */
   int select_j_rand(int i);
  /**
   * @brief limite the label between H & L
   * 
   */
   int clipAlpha(int label , int H , int L);
  /**
   * @brief updata error of number_k
   * 
   */
  int updata_EK(int k);
};


