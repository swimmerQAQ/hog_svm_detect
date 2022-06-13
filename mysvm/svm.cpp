#include "svm.h"
#include <fstream>
#include <Eigen/src/Core/Map.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
int MYSVM::_init_SVM(Eigen::MatrixXd datain , Eigen::MatrixXd labels , float c , float toler)
{
    this->_data = datain;
    this->_label = labels;
    this->_C = c;
    this->_tol = toler;
    this->_data_num = int(datain.rows());
    this->_b = 0;

    // cout << _label << endl;
    
    this->Kernel = Eigen::MatrixXd::Zero(datain.rows(),datain.rows());
    this->_alphas = Eigen::VectorXd::Zero(datain.rows(),1);
    this->_error_box = Eigen::MatrixXd::Zero(datain.rows(),2);
    return 1;
}
Eigen::MatrixXd openData(string fileToOpen)
{
    vector<double> matrixEntries;
    ifstream matrixDataFile(fileToOpen);
    string matrixRowString;
    string matrixEntry;
    int matrixRowNumber = 0;
    if (!matrixDataFile.is_open())
    {
        cout << " Cannot open data file. " << endl;
    }
    // cout << fileToOpen << endl;
    while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        // cout << matrixRowString << endl;
        stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.
 
        while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            // cout << matrixEntry << endl;
            // stod() input string and return a double number
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }
    // for (auto it : matrixEntries)
    // {
    //     cout << it << endl;
    // }
    // here we convet the vector variable into the matrix and return the resulting object, 
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    // return Eigen::MatrixXd();
}
int MYSVM::load_file(string filepath)
{
    auto data_label = openData(filepath);
    int rows = data_label.rows();
    int cols = data_label.cols();
    // cout  << "row =" << rows << " col =" << cols << endl;
    Eigen::MatrixXd data = data_label.block(0,0,rows,(cols-1));
    // cout << data << endl;
    //from (o,cols-1) to x -> rows and y -> 1
    Eigen::MatrixXd label = data_label.block(0,(cols-1),rows,1);
    // cout << label << endl;
    _init_SVM(data,label,1,1);
    this->Kernel = radial_bassic_func(data,1);
    // for (int i = 0; i  < 100 ; i++) /////////////test for trace()
    // all the num of kernel(i,i) is 1, which means e^(Xi - Xi) = 1 
    // {
    //     /* code */
    //     for (int j = 0; j  < 100 ; j++)
    //     {
    //         if (i == j)
    //         {
    //             cout << Kernel(i,j) << endl;
    //         }
    //     }
    // }
    _afterInit = true;
    return 1;
}
Eigen::MatrixXd MYSVM::radial_bassic_func(Eigen::MatrixXd data , float k)
{
    Eigen::MatrixXd count_kernel = Eigen::MatrixXd(data.rows(),data.rows());
    // cout << data << endl;
    for (int i = 0; i < data.rows() ; i ++)
    {
        for (int j = 0; j < data.rows(); j++)
        {
            /* kernel */
            count_kernel(j,i) = (data.row(i) - data.row(j)) * (data.row(i) - data.row(j)).transpose();
            float temp = count_kernel(j,i);
            count_kernel(j,i) = std::exp(temp/(-1*pow(k,2)));
        }
         
    }
    return count_kernel;
}
int MYSVM::smo(int max_for)
{
    if (!_afterInit)
    {
        cout << " error prepare for smo " << endl;
        return -1;
    }
    /**
     * @brief smo循环条件 
     * 达到最大迭代次数    且alpha值没有改变 -> 退出
     * 或是遍历所有数据    且alpha值没有改变 -> 退出
     * 
     */
    //迭代次数
    int num_circle = 0;
    //alpha对的改变情况： =0 means have some changes
    int alphaPairs_Change = 0;
    //遍历所有的数据 true means 没有遍历数据
    bool entire_Set= true;
    innerL(1);
    // while ( (alphaPairs_Change>0 && num_circle > max_for) || (entire_Set) )
    // {
    //     /* 重新载入一次遍历时将先alpha置0，用于这一次遍历的记录 */
    //     alphaPairs_Change = 0;
    //     if (entire_Set)
    //     {//没有遍历数据
    //         for (int i = 0; i < this->_data_num ; i++)
    //         {
    //             alphaPairs_Change += innerL(i);
    //         }

    //     }
    // }
    


    return 1;
}
int bigger(int one , int two)
{
    if (one > two)
    {
        return one;
    }
    else
    {
        return two;
    }
    return one;
}
int smaller(int one , int two)
{
    if (one < two)
    {
        return one;
    }
    else
    {
        return two;
    }
    return one;
}
int MYSVM::updata_EK(int k)
{
    float Ek = calculate_error(k);
    this->_error_box(k,0) = 1;
    this->_error_box(k,1) = Ek;
    return Ek;
}
int MYSVM::innerL(int ord)
{
    float Errori = calculate_error(ord);
    bool slack_error_rate = (_label(ord,0)*Errori < -1*_tol)&&(_alphas(ord,0) < _C);
    bool occurr_the_alpha = (_label(ord,0)*Errori > _tol) && (_label(ord,0) > 0);
    if (slack_error_rate || occurr_the_alpha)
    {
        Eigen::Vector2f JEJ = select_j(ord,Errori);
        float alpha_i_old = this->_alphas(ord,0);//old copy the i and j labels
        float alpha_j_old = this->_alphas(JEJ(0),0); 
        int L = 0,H = 0;
        if (_label(ord,0) != _label(JEJ(0),0))
        {
            L = bigger(0,_label(ord,0) - _label(JEJ(0),0));
            H = smaller(_C,_C + _label(ord,0) - _label(JEJ(0),0));
        }
        else
        {
            L = bigger(0,_label(ord,0) + _label(JEJ(0),0) - _C);
            H = smaller(_C,_label(ord,0) + _label(JEJ(0),0));
        }
        if (L == H)//出现 上下限一样的情况
        {
            return 0;
        }
        float temp1 = _data.row(ord) * _data.row(JEJ[0]).transpose();
        float temp2 = _data.row(ord) * _data.row(ord).transpose();
        float temp3 = _data.row(JEJ[0]) * _data.row(JEJ[0]).transpose();
        float eta = 2*temp1 - temp2 - temp3;
        if (eta >= 0)//出现eta大于0？？？
        {
            return 0;
        }
        //updata the alpha_j
        _alphas(JEJ(0),0) -= _label(JEJ(0),0) * (Errori - JEJ(1))/eta;
        //limit the _alpha(JEJ(0))
        _alphas(JEJ(0),0) = clipAlpha(_label(JEJ(0),0) , H , L);
        updata_EK(JEJ[0]);
        double delta_j_alpha = _alphas(JEJ[0],0) - alpha_j_old;
        if (abs(delta_j_alpha) < 0.00001)
        {
            return 0;
        }
        _alphas(ord,0) += _alphas(JEJ[0],0) * _alphas(ord,0)*( alpha_j_old - _alphas(JEJ[0],0) );

        //update b1 and b2
        float b1 = _b - Errori - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp2 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp2;
        float b2 = _b - JEJ[1] - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp1 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp3;
        if (0 < _alphas(ord,0) && _C > _alphas(ord,0))
        {
            _b = b1;
        }
        else if (0 < _alphas(JEJ[0],0) && _C > _alphas(JEJ[0],0))
        {
            _b = b2;
        }
        else
        {
            _b = (b1 + b2)/2.0;
        }
        return 1;
    }
    return 0;
}
//matrix
/**
 * @brief vector multiply
 * 
 * @param one a vector -> alpha
 * @param two a vector -> label
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd dot(Eigen::MatrixXd one , Eigen::MatrixXd two)
{
    if (one.rows() != two.rows())
    {
        cout << "error dot点乘 " << endl;
    }
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(one.rows(),1);
    for (int i = 0; i < one.rows(); i++)
    {
        temp(i,0) = one(i,0)*two(i,0);
    }
    return temp;
    
}
float MYSVM::calculate_error(int ord)
{
    Eigen::MatrixXd temp = dot(_alphas,_label).transpose()*Kernel.col(ord);
    float Error_ord = temp.sum() - this->_label(ord,0);
    cout << Error_ord;
    return Error_ord;
}

Eigen::Vector2f MYSVM::select_j(int i, float Ei)
{
    float maxError = 0;
    Eigen::Vector2f JEJ = Eigen::Vector2f::Zero();
    this->_error_box(i,0) = 1;
    this->_error_box(i,1) = Ei;
    Eigen::VectorXf error_box_valid_list = valid_list(_error_box);
    if (error_box_valid_list.size() != 0)
    {
        for (int P=0 ; P < error_box_valid_list.size() ; P ++)
        {
            if (error_box_valid_list(P) == i)
            {
                continue;
            }
            float Ek = calculate_error(P);
            float deltaE = abs(Ei - Ek);
            if (deltaE > maxError)
            {
                JEJ(0) = P;
                maxError = deltaE;
                JEJ(1) = Ek;
            }
        }
        return JEJ;
    }
    else
    {
        JEJ(0) = select_j_rand(i);
        JEJ(1) = calculate_error(JEJ(0));
    }
    return JEJ;
}
Eigen::VectorXf MYSVM::valid_list(Eigen::MatrixXd error_box)
{
    int rows = error_box.rows();
    Eigen::VectorXf list;
    for (int i = 0 ; i < error_box.rows() ; i++)
    {
        if (error_box(i,0) != 0)
        {
            list << i;
        }
    }
    return list;
}
double random(double start , double end)
{
    return start + (end - start)*rand()/(RAND_MAX + 1.0);
}
int MYSVM::select_j_rand(int i)
{
    int j = i;
    while (j == i)
    {
        /* code */
        srand(unsigned(time(0)));
        j = int(random(0,_data_num));
    }
    return j;
}
int MYSVM::clipAlpha(int label , int H , int L)
{
    if (label > H)
    {
        label = H;
    }
    if (L > label)
    {
        label = L;
    }
    return label;
}