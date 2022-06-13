#include <opencv2/opencv.hpp>
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
    std::vector<int> _angle_feature;
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
    int h_features(void);
    int v_features(void);
};
