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
    /*data and label*/
    auto data_label = openData(filepath);
    int rows = data_label.rows();
    int cols = data_label.cols();
    // cout  << "row =" << rows << " col =" << cols << endl;
    Eigen::MatrixXd data = data_label.block(0,0,rows,(cols-1));
    // cout << data << endl;
    //from (o,cols-1) to x -> rows and y -> 1
    Eigen::MatrixXd label = data_label.block(0,(cols-1),rows,1);
    // cout << label << endl;
    _init_SVM(data,label,200,0.0001);
    this->Kernel = radial_bassic_func(data,1.3);
    // for (int i = 0; i  < 100 ; i++) /////////////test for trace()
    // all the num of kernel(i,i) is 1, which means e^(Xi - Xi) = 1 
    {
        /* code */
        // for (int j = 0; j  < 100 ; j++)
        // {
        //     // if (j == 50)
        //     {
        //         cout << Kernel << endl;
        //     }
        // }
    }
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
            count_kernel(j,i) = (data.row(j) - data.row(i)) * (data.row(j) - data.row(i)).transpose();
            double temp = count_kernel(j,i);
            count_kernel(j,i) = std::exp(temp/(-1*pow(k,2)));
            // if (i == 10 && j == 15)
            // {
            //     cout << " data " << data.row(j) << data.row(i) << endl;
            //     cout << " result " << count_kernel(j,i) << endl;
            // }
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
    // innerL(1);
    while ( num_circle < max_for && (alphaPairs_Change>0 || (entire_Set)) )
    {
        /* 重新载入一次遍历时将先alpha置0，用于这一次遍历的记录 */
        alphaPairs_Change = 0;
        /*第一次遍历所有的样本*/
        if (entire_Set)
        {//没有遍历数据
            for (int i = 0; i < this->_data_num ; i++)
            {
                alphaPairs_Change += innerL(i);
                cout << "search in all data. epoch:"<< num_circle << " , alpha_changes:" << alphaPairs_Change << endl;
            }
            num_circle++;
        }
        else
        {
            Eigen::VectorXd list_none_zero_alpha = no_zero(_alphas);
            // cout << "non zero" << list_none_zero_alpha << endl;
            for (int i =0 ; i < list_none_zero_alpha.size() ; i++)
            {
                alphaPairs_Change += innerL(int(list_none_zero_alpha(i)));
                cout << "search in alpha!=0; epoch:" << num_circle << " , alpha_changes:" << alphaPairs_Change<< endl;
            }
            num_circle++;
        }
        /*如果遍历过整个数据，设置标志已遍历过*/
        if (entire_Set)
        {
            entire_Set = false;
        }
        /*没有超过最大次数时，全遍历但是一个alpha都没有改变直接离开（为便利或遍历有效才继续while*/
        else if (alphaPairs_Change == 0)/*设置如果已经全遍历过之后，出现没有一个alpha对改变的情况，设置重新遍历， 配合前一条生成或条件开关*/
        {
            entire_Set = true;
        }
    }
    return 1;
}
double bigger(double one , double two)
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
double smaller(double one , double two)
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
    double Errori = calculate_error(ord);
    bool slack_error_rate = (_label(ord,0)*Errori < -1*_tol)&&(_alphas(ord,0) < _C);
    bool occurr_the_alpha = (_label(ord,0)*Errori > _tol) && (_alphas(ord,0) > 0);
    if (slack_error_rate || occurr_the_alpha)
    {
        Eigen::Vector2d JEJ = select_j(ord,Errori);
        double alpha_i_old = this->_alphas(ord,0);//old copy the i and j labels
        double alpha_j_old = this->_alphas(JEJ(0),0); 
        double L = 0, H = 0;
        // cout << _label(ord,0)  << "select" << _label(JEJ(0),0) << endl;
        if (_label(ord,0) != _label(JEJ(0),0))
        {
            L = bigger(0,_alphas(JEJ(0),0) - _alphas(ord,0));
            H = smaller(_C,_C + _alphas(JEJ(0),0) - _alphas(ord,0));
        }
        else
        {
            L = bigger(0,_alphas(ord,0) + _alphas(JEJ(0),0) - _C);
            H = smaller(_C,_alphas(ord,0) + _alphas(JEJ(0),0));
        }
        // cout << H << "limit ??" << L << endl;
        if (L == H)//出现 上下限一样的情况
        {
            return 0;
        }
        double temp1 = Kernel(ord , JEJ[0]);
        double temp2 = Kernel(ord , ord);
        double temp3 = Kernel(JEJ[0] , JEJ[0]);
        double eta = 2*temp1 - temp2 - temp3;
        if (eta >= 0)//出现eta大于0？？？
        {
            return 0;
        }
        //updata the alpha_j
        // cout << _label(JEJ(0),0) * (Errori - JEJ(1))/eta << endl;
        // cout << _alphas(JEJ(0),0) << endl;
        _alphas(JEJ(0),0) -= _label(JEJ(0),0) * (Errori - JEJ(1))/eta;
        // cout << " after change " << _alphas(JEJ(0),0) << endl;
        //limit the _alpha(JEJ(0))

        _alphas(JEJ(0),0) = clipAlpha(_alphas(JEJ(0),0) , H , L);
        updata_EK(JEJ[0]);
        double delta_j_alpha = _alphas(JEJ[0],0) - alpha_j_old;
        if (abs(delta_j_alpha) < 0.00001)
        {
            return 0;
        }
        _alphas(ord,0) += _label(JEJ[0],0) * _label(ord,0)*( alpha_j_old - _alphas(JEJ[0],0) );
        // if (abs(_alphas(ord,0)) < 0.00001)
        // {
        //     _alphas(ord,0) = 0;
        // }
        updata_EK(ord);
        //update b1 and b2
        double b1 = _b - Errori - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp2 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp2;
        double b2 = _b - JEJ[1] - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp1 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp3;
        if ( (0 < _alphas(ord,0)) && (_C > _alphas(ord,0)) )
        {
            _b = b1;
        }
        else if ( (0 < _alphas(JEJ[0],0)) && (_C > _alphas(JEJ[0],0)) )
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
double MYSVM::calculate_error(int ord)
{
    Eigen::MatrixXd temp = dot(_alphas,_label).transpose()*Kernel.col(ord);
    double Error_ord = temp.sum() + _b - _label(ord,0);
    // cout << "calculate error " << Error_ord << endl;
    // cout  << ": " << Error_ord << " " << _label(ord,0) << endl;
    return Error_ord;
}
Eigen::Vector2d MYSVM::select_j(int i, double Ei)
{
    float maxError = 0;
    Eigen::Vector2d JEJ = Eigen::Vector2d::Zero();
    this->_error_box(i,0) = 1;
    this->_error_box(i,1) = Ei;
    // cout << "error_box " << _error_box << endl;
    Eigen::VectorXd error_box_valid_list = valid_list(_error_box);
    // cout << "valid " << error_box_valid_list << endl;
    // cout << "valid error list" << error_box_valid_list << endl;
    if (error_box_valid_list.size() != 0)
    {
        for (int P=0 ; P < error_box_valid_list.size() ; P ++)
        {
            int valid_num = error_box_valid_list(P);
            // cout << "validnum" << valid_num << "i = " << i << endl;
            if (valid_num == i)
            {
                continue;
            }
            double Ek = calculate_error(valid_num);
            double deltaE = abs(Ei - Ek);
            if (deltaE > maxError)
            {
                JEJ(0) = valid_num;
                maxError = deltaE;
                JEJ(1) = Ek;
            }
            // cout << "maxerror j Ej" << JEJ[0] << " " << JEJ[1] << " " << maxError << endl;
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
Eigen::VectorXd MYSVM::valid_list(Eigen::MatrixXd error_box)
{
    int rows = error_box.rows();
    Eigen::VectorXd list;
    int number = 0;
    for (int i = 0 ; i < error_box.rows() ; i++)
    {
        if (error_box(i,0) != 0)
        {
            // list << i;
            number++;
        }
    }
    list.resize(number);
    for (int i = 0 ; i < error_box.rows() ; i++)
    {
        if (error_box(i,0) != 0)
        {
            list(number-1) = i;
            number--;
        }
    }
    return list;
}
Eigen::VectorXd MYSVM::no_zero(Eigen::MatrixXd alphas)
{
    int rows = alphas.size();
    Eigen::VectorXd list;
    int number = 0;
    for (int i = 0 ; i < rows ; i++)
    {
        if (alphas(i) != 0 && alphas(i) < _C)
        {
            number++;
        }
    }
    list.resize(number);
    for (int i = 0 ; i < rows ; i++)
    {
        if (alphas(i) != 0 && alphas(i) < _C)
        {
            list(number-1) = i;
            number--;
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
    cout << j << endl;
    return j;
}
double MYSVM::clipAlpha(double label , double H , double L)
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