#include "hog_feature.h"
#include <iostream>
#include <fstream>
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
    _angle_feature.clear();
    _angle_feature.resize((_origin.rows - 2) * (_origin.cols - 2));
    // cout << (_origin.rows - 2)  << (_origin.cols - 2) <<endl;
    return 1;
}
int strengthen_feature(int one , int two)
{
    int temp = abs(one - two);
    return (2*temp - double(pow(temp,2))/255.0);
}
int HOG_FEATURE::pre_feature(void)
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
                          uchar *green_bin = _green_MODULES_x.ptr<uchar>(row);
                          uchar *data_src = _origin.ptr<uchar>(row);
                          for (int col = 0+1; col < _origin.cols-1; col++)
                          { //蓝-绿-红
                            green_bin[col] = sqrt(pow(data_src[3*col + 0],2)  + pow(data_src[3*col + 1],2) + pow(data_src[3*col + 2],2));
                            // green_bin[col] = int(3*green_bin[col] - double(3*pow(green_bin[col],3))/pow(255.0,2));
                            if (green_bin[col] < 110)
                            {
                                green_bin[col] = 0;
                            }else{
                                green_bin[col] = 255;
                            }
                            }
                        }   
                  }
                    );
    cv::imshow(" fliter ",_green_MODULES_x);
    return 1;
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
                                // blue_bin[col] = strengthen_feature(int(data_src[3 * down + 0]) , int(data_src[3 * up + 0]));
                                blue_bin[col] = strengthen_feature(green_bin[down] , green_bin[up]);
                                // green_bin[col] = strengthen_feature(int(data_src[3 * down + 1]) , int(data_src[3 * up + 1]));
                                // _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] = data_src[3 * down + 0] - int(data_src[3 * up + 0]);
                                _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] = green_bin[down] - green_bin[up];
                                
                                orient_bin[col] = blue_bin[col];//red_bin[col];
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
    // for (auto it : _angle_feature)
    // {
    //     if (it == 1)
    //     {
    //         continue;
    //     }
    //     cout << " angle =  " << it  << endl;
    // }
    // cv::imshow("hog_blue",_blue_MODULES_x);
    // cv::imshow("hog_red",_red_MODULES_x);
    // cv::imshow("hog_green",_green_MODULES_x);
    return 1;
}
int module(int one , int two)
{
    // return (one + two)/2;
    return sqrt(pow(one,2)+pow(two,2));
}
int HOG_FEATURE::v_features(void)
{
    if (_origin.size == 0)
    {
        cout << " size invaild in getting features... in HOG::features(void); " << endl;
    }
    cv::parallel_for_(cv::Range(1, _origin.cols-1),
                  [&](const cv::Range &range)
                  {
                      for (int col = range.start; col < range.end; col++)
                      {
                        for (int row = 0+1; row < _origin.rows-1; row++)
                          { //蓝-绿-红
                            int up = row + 1;
                            int down = row - 1;
                            int vertical = 0;
                            
                            // vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 0] , _origin.ptr<uchar>(down)[ 3 * col + 0]);
                            vertical = strengthen_feature(_green_MODULES_x.ptr<uchar>(up)[col] , _green_MODULES_x.ptr<uchar>(down)[col]);
                            _orientention.ptr<uchar>(row)[col] = module(vertical,_orientention.ptr<uchar>(row)[col]);
                            // _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] = atan2(_origin.ptr<uchar>(up)[ 3 * col + 0] - _origin.ptr<uchar>(down)[ 3 * col + 0],_angle_feature[(_origin.cols -2) * (row - 1) + col - 1])/(2*acos(-1))*360;
                            _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] = atan2(_green_MODULES_x.ptr<uchar>(up)[col] - _green_MODULES_x.ptr<uchar>(down)[col],_angle_feature[(_origin.cols -2) * (row - 1) + col - 1])/(2*acos(-1))*360;
                            if (_angle_feature[(_origin.cols -2) * (row - 1) + col - 1] < -87 && _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] > -93)
                            // if (abs(_angle_feature[(_origin.cols -2) * (row - 1) + col - 1]) > 10)
                            {
                                // cout << "angle = : " << _angle_feature[( col - 1 ) * (_origin.rows - 2) + row - 1 ] << endl;
                                _red_MODULES_x.ptr<uchar>(row)[col] = 255;
                                _origin.ptr<uchar>(down)[ 3 * col + 1] = 255;
                                _origin.ptr<uchar>(down)[ 3 * col + 2] = 0;
                                _origin.ptr<uchar>(down)[ 3 * col + 0] = 0;
                            }
                            /* green */
                            // vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 1] , _origin.ptr<uchar>(down)[ 3 * col + 1]);
                            // _angle_feature[down * (_origin.cols - 2) + col - 1] = atan2(vertical,_angle_feature[down * (_origin.cols - 1) + col - 1])/(2*acos(-1))*360;

                            /* red */
                            // vertical = strengthen_feature(_origin.ptr<uchar>(up)[ 3 * col + 2] , _origin.ptr<uchar>(down)[ 3 * col + 2]);

                            // cout << " out of temp in cols " << endl;
                            // cout << " :" << _orientention.ptr<uchar>(row)[col] - 0 << " next: ";
                            }
                        }   
                  }
                    );
    // cout << _angle_feature.size() << "is size " << _orientention.size();
    cv::imshow("origin" , _origin);
    cv::imshow("angle" , _red_MODULES_x);
    cv::imshow("orientention",_orientention);
    // cv::imwrite("./origin.jpg",_origin);
    // cv::imwrite("./45angle.jpg",_red_MODULES_x);
    // cv::imwrite("./orientention.jpg",_orientention);
    return 1;
}
int HOG_FEATURE::scrap_frame(void)
{
    if (_origin.size == 0)
    {
        cout << " size invaild in getting features... in HOG::features(void); " << endl;
    }
    _hog_box.clear();
    _hog_box.resize(10);
    for (int i =0 ; i < _hog_box.size() ; i++)
    {
        _hog_box[i].resize(10,Eigen::MatrixXd::Zero(9,1));
        // cout << "size" << hog.size() << " 1 " << hog[1] << endl;
    }
    // cout << " hog_box " << _hog_box[0][0] << endl;
    cv::parallel_for_(cv::Range(1, _origin.cols-1),
                  [&](const cv::Range &range)
                  {
                      for (int col = range.start; col < range.end; col++)
                      {
                        int num_col = col/16;
                        for (int row = 0+1; row < _origin.rows-1; row++)
                          { //蓝-绿-红
                            int num_row = row/16;
                            // _angle_feature[(_origin.cols -2) * (row - 1) + col - 1] and _orientention.ptr<uchar>(row)[col]
                            int hog_num = abs(_angle_feature[(_origin.cols -2) * (row - 1) + col - 1])/20;
                            if (hog_num == 9)
                            {
                                hog_num = 8;
                            }
                            // cout << hog_num << endl;
                            _hog_box[num_row][num_col](hog_num,0) = _hog_box[num_row][num_col](hog_num,0)+ _orientention.ptr<uchar>(row)[col];
                            }
                        }   
                  }
                    );
    return 1;
}
void saveData(string fileName, Eigen::MatrixXd  matrix)
{
	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix << '\n';
		file.close();
	}
}
Eigen::MatrixXd maxmin(Eigen::MatrixXd data)
{
    for (size_t i = 0; i < data.cols()-1; i++)
    {
        double max = data.col(i).maxCoeff();
        double min = data.col(i).minCoeff();
        for (int j =0 ; j < data.col(i).size(); j ++ )
        {
            if (max - min == 0)
            {
                data.col(i)[j] = 0;
                continue;
            }
            data.col(i)[j] = (data.col(i)[j] - min) / (max - min);
        }
    }
    return data;

}
void HOG_FEATURE::test_show_hog(string filename, int num)
{
    for (size_t i = 0; i < _hog_box.size(); i++)
    {
        for (size_t j = 0; j < _hog_box[i].size(); j++)
        {
            // std::cout << " hog feature " << "i = " << i << " j = " << j << std::endl << _hog_box[i][j] << std::endl << std::endl;
            for (size_t k = 0; k < _hog_box[i][j].size(); k++)
            {
                _base(num, i*9*10 + j*9 + k) = _hog_box[i][j](k,0);
            }
        }
    }
    //标签位置
    _base(num, 900) = -1;
    // if (num == 10)
    // {
    //     for (size_t i = 0; i < _hog_box.size(); i++)
    // {
    //     for (size_t j = 0; j < _hog_box[i].size(); j++)
    //     {
    //         // std::cout << " hog feature " << "i = " << i << " j = " << j << std::endl << _hog_box[i][j] << std::endl << std::endl;
    //         for (size_t k = 0; k < _hog_box[i][j].size(); k++)
    //         {
    //            cout <<  _hog_box[i][j](k,0) << " ";
    //         }
    //     }
    // }
    // cout << endl;
    // }
    if (_train_num == num +1)
    {
        // cout << " pre " <<_base.col(900) << endl;
        _base = maxmin(_base);
        // cout << _base.col(900) << endl;
        saveData(filename,_base);
    }

}
