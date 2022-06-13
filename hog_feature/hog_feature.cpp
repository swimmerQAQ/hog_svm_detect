#include "hog_feature.h"
#include <iostream>
using namespace std;
int HOG_FEATURE::load_image(cv::Mat &input_image)
{
    if ( input_image.size == 0)
    {
        cout << " Input invaild image, which size is zero. " << endl;
        return -1;
    }
    _origin = input_image;
    _red_MODULES_x = cv::Mat::zeros(_origin.size(), CV_8UC1);
    _blue_MODULES_x = cv::Mat::zeros(_origin.size(), CV_8UC1);
    _green_MODULES_x = cv::Mat::zeros(_origin.size(), CV_8UC1);
    _orientention = cv::Mat::zeros(_origin.size(), CV_8UC1);
    _angle_feature.resize((_origin.rows - 2) * (_origin.cols - 2));
    
    cout << (_origin.rows - 2)  << (_origin.cols - 2) <<endl;
    return 1;
}
int strengthen_feature(int one , int two)
{
    int temp = abs(one - two);
    return (2*temp - pow(temp,2)/255);
}
int HOG_FEATURE::h_features(void)
{
    if (_origin.size == 0)
    {
        cout << " size invaild in getting features... in HOG::features(void); " << endl;
    }
    cv::parallel_for_(cv::Range(1, _origin.rows-1),
                  [&](const cv::Range &range)
                  {
                      for (int row = range.start; row < range.end; row++)
                      {
                          uchar *data_src = _origin.ptr<uchar>(row);
                          uchar *red_bin = _red_MODULES_x.ptr<uchar>(row);
                          uchar *blue_bin = _blue_MODULES_x.ptr<uchar>(row);
                          uchar *green_bin = _green_MODULES_x.ptr<uchar>(row);
                          uchar *orient_bin = _orientention.ptr<uchar>(row);
                          for (int col = 0+1; col < _origin.cols-1; col++)
                          { //蓝-绿-红
                                
                                int up = col - 1;
                                int down = col + 1;
                                red_bin[col] = strengthen_feature(int(data_src[3 * down + 2]) , int((data_src[3 * up + 2])));
                                blue_bin[col] = strengthen_feature(int(data_src[3 * down + 0]) , int(data_src[3 * up + 0]));
                                green_bin[col] = strengthen_feature(int(data_src[3 * down + 1]) , int(data_src[3 * up + 1]));
                                _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] = blue_bin[col];
                                // orient_bin[col] = 100;//red_bin[col];
                                // if (blue_bin[col] > orient_bin[col])
                                // {
                                //     orient_bin[col] = blue_bin[col];
                                // }
                                // if (green_bin[col] > orient_bin[col])
                                // {
                                //     orient_bin[col] = green_bin[col];
                                // }
                            }   
                        }   
                  }
                    );
    cv::imshow("hog_blue",_blue_MODULES_x);
    cv::imshow("hog_red",_red_MODULES_x);
    cv::imshow("hog_green",_green_MODULES_x);
    return 1;
}
int module(int one , int two)
{
    return sqrt(pow(one,2)+pow(two,2));
}
int HOG_FEATURE::v_features(void)
{
    if (_origin.size == 0)
    {
        cout << " size invaild in getting features... in HOG::features(void); " << endl;
    }
    cv::parallel_for_(cv::Range(1, _origin.cols-2),
                  [&](const cv::Range &range)
                  {
                      for (int col = range.start; col < range.end; col++)
                      {
                        int temp = col%3;
                        for (int row = 0+1; row < _origin.rows-1; row++)
                          { //蓝-绿-红
                            int up = row + 1;
                            int down = row - 1;
                            int vertical = 0;
                            switch (temp)
                            {
                            case 0:
                            vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 0] , _origin.ptr<uchar>(down)[ 3 * col + 0]);
                            _orientention.ptr<uchar>(row)[col] = module(vertical,_angle_feature[down * (_origin.cols - 2) + col - 1]);
                            _angle_feature[down * (_origin.cols - 2) + col - 1] = atan2(vertical,_angle_feature[down * (_origin.cols - 2) + col - 1])/(2*acos(-1))*360;
                            break;
                            case 1:
                            vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 1] , _origin.ptr<uchar>(down)[ 3 * col + 1]);
                            _angle_feature[down * (_origin.cols - 2) + col - 1] = atan2(vertical,_angle_feature[down * (_origin.cols - 2) + col - 1])/(2*acos(-1))*360;
                            break;
                            case 2:
                            vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 2] , _origin.ptr<uchar>(down)[ 3 * col + 2]);
                            _angle_feature[down * (_origin.cols - 2) + col - 1] = atan2(vertical,_angle_feature[down * (_origin.cols - 2) + col - 1])/(2*acos(-1))*360;
                            break;
                            default:
                            cout << " out of temp in cols " << endl;
                            break;
                            }
                            // cout << " :" << _orientention.ptr<uchar>(row)[col] - 0 << " next: ";
                            }
                        }   
                  }
                    );
    // cout << _angle_feature.size() << "is size " << _orientention.size();
    // for (auto it : _angle_feature)
    // {
    //     cout << it ;
    // }
    cv::imshow("orientention",_orientention);
    return 1;
}