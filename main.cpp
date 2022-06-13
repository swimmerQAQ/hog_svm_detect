#include <opencv2/opencv.hpp>
#include <iostream>
#include "./hog_feature/hog_feature.h"
#include "./mysvm/svm.h"
using namespace std;
string path = "/home/ksc/Documents/Project_Base/hog_feature/test.avi";
int main()
{
    cv::VideoCapture capture(path);
    cv::Mat one_frame;
    if (!capture.isOpened())
    {
        cout << " Fail to open camera... " << endl;
        return 0;
    }
    
    HOG_FEATURE myhog;
    // while (1)
    // {
    //     capture >> one_frame;
    //     cv::resize(one_frame,one_frame,cv::Size(500,500));
    //     // cv::imshow("hog",one_frame);
    //     myhog.load_image(one_frame);
    //     myhog.h_features();
    //     myhog.v_features();
    //     if (cv::waitKey(30) == 32)
    //     {
    //         if (cv::waitKey(0) == 27)
    //         {
    //             break;
    //         }
    //     }
    // }
    MYSVM mysvm;
    mysvm.load_file("../testdata1.csv");
    mysvm.smo(100);
    return 0;
}