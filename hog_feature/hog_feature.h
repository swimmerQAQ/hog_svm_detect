#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
class HOG_FEATURE
{
private:
    const int PI = 4*atan(1);
    /**
     * @brief origin image
     */
    cv::Mat _origin;
    /**
     * @brief modules features
     */
    cv::Mat _red_MODULES_x;
    cv::Mat _blue_MODULES_x;
    cv::Mat _green_MODULES_x;
    cv::Mat _orientention;
    /**
     * @brief angle features
     * 
     */
    std::vector<double> _angle_feature;
    /**
     * @brief count hog
     * 
     */
    std::vector<std::vector<Eigen::MatrixXd>> _hog_box;
    /**
     * @brief train_set num
     * 
     */
    int _train_num;
    /**
     * @brief base
     * 
     */
    Eigen::MatrixXd _base;
public:
    HOG_FEATURE(/* args */){};
    ~HOG_FEATURE(){};
    /**
     * @brief load a imgae
     */
    int load_image(cv::Mat &input_image);

    /**
     * @brief find features
     * @brief vertical horizontal
     */
    int pre_feature(void);
    int h_features(void);
    int v_features(void);
    /**
     * @brief scrap_frame into 16 frames
     * 
     */
    int scrap_frame(void);
    /**
     * @brief test show scrap_frame
     * 
     */
    void test_show_hog(std::string filename, int num);
    /**
     * @brief num of train set
     * 
     */
    void train_set(int num){_train_num = num;_base = Eigen::MatrixXd::Zero(_train_num,901);};
};
