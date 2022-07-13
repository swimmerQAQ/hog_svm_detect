#include <opencv2/opencv.hpp>
#include <iostream>
#include "./hog_feature/hog_feature.h"
#include "./mysvm/svm.h"
using namespace std;
string path = "/home/ksc/Documents/Project_Base/hog_feature/test.avi";

int main()
{
    // cv::VideoCapture capture(path);
    // cv::Mat one_frame = cv::imread("/home/ksc/Documents/Project_Base/hog_feature/Sample001/img001-001.png");

    
    /*--------------------------取消该段注释查看hog特征----------------------------*/
    /* pictures */
    string folder_path = "../4number";
    std::vector<cv::String> file_names;
    cv::glob(folder_path, file_names);   //get file names
    /**/ 

    // if (!capture.isOpened())
    // {
    //     cout << " Fail to open camera... " << endl;
    //     return 0;
    // }
    
    HOG_FEATURE myhog;
    myhog.train_set(file_names.size());
    cv::Mat img;
    //img = cv::imread("../5number/65.jpg");
    // while (1)
    for (size_t i = 0; i < file_names.size(); i++)
    {
        img = cv::imread(file_names[i]);
        if (!img.data)
        {
            continue;
        }
        
    //     // capture >> one_frame;
        cv::resize(img,img,cv::Size(160,160));
        cv::imshow("number img",img);
        // cv::imshow("hog",one_frame);
        myhog.load_image(img);
        myhog.pre_feature();
        myhog.h_features();
        myhog.v_features();
        myhog.scrap_frame();
        myhog.test_show_hog("../numberfour_.csv", i);
        // cv::waitKey(0);
        if (cv::waitKey(1000) == 32)
        {
            if (cv::waitKey(0) == 27)
            {
                break;
            }
        }
    }
    /*--------------------------------------------------------------------------*/


    /*----------------------------取消该段注释并配置数据文件，可以使用svm分类器测试对应------------------------------*/
    /**
     * @brief 输出范例： 把第 99 个样本正确分类了。。。。应该是 -1 分类样本是 type > 0  : type_value
     * 
     */
    // MYSVM mysvm;
    // mysvm.load_file("../tra5.csv");
    // mysvm.smo(200);
    // // cout << "当前是 数字" << i << " 的alpha 和 b " << endl; 
    // // mysvm.show_somthing();
    // mysvm.test("../tes5.csv");
    /*-----------------------------------------------------------------------------------------------------*/
    return 0;
}