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
Eigen::MatrixXd maxmin(Eigen::MatrixXd data)
{
    Eigen::MatrixXd temp = Eigen::MatrixXd(data.rows(),data.cols());
    for (size_t i = 0; i < data.cols(); i++)
    {
        double max = data.col(i).maxCoeff();
        double min = data.col(i).minCoeff();
        for (int j =0 ; j < data.col(i).size(); j ++ )
        {
            if (max - min == 0)
            {
                temp.col(i)[j] = 0;
                continue;
            }
            temp.col(i)[j] = (data.col(i)[j] - min) / (max - min);
        }
    }
    return temp;

}
int MYSVM::load_file(string filepath)
{
    double temp = 0.0001;
    // if(filepath == "../svmtrain.csv")
    // {
    //     flag = false;
    //     temp = 0.0001;
    // }
    /*data and label*/
    auto data_label = openData(filepath);
    int rows = data_label.rows();
    int cols = data_label.cols();
    // cout  << "row =" << rows << " col =" << cols << endl;
    Eigen::MatrixXd data = data_label.block(0,0,rows,(cols-1));
    // cout << data << endl;
    //from (o,cols-1) to x -> rows and y -> 1
    // data = maxmin(data);
    // cout << data.row(50);
    Eigen::MatrixXd label = data_label.block(0,(cols-1),rows,1);
    // cout << label << endl;
    _init_SVM(data,label,200,temp);
    this->Kernel = radial_bassic_func(data,1.3);
    // for (int i = 0; i  < 60 ; i++) /////////////test for trace()
    // // all the num of kernel(i,i) is 1, which means e^(Xi - Xi) = 1 
    // {
    //     /* code */
    //     for (int j = 0; j  < 60 ; j++)
    //     {
    //         // if (j == i)
    //         {
    //             cout  << " kernel " << Kernel(i,j) << endl;
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
            count_kernel(j,i) = (data.row(j) - data.row(i)) * (data.row(j) - data.row(i)).transpose();
            double temp = count_kernel(j,i);
            count_kernel(j,i) = std::exp(temp/(-1*pow(k,2)));
            if (i == 10 && j == 15)
            {
                // cout << " data " << data.row(j) << " another " << data.row(i) << endl;
                // cout << " result "  << temp << endl;
            }
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
                // cout << "search in all data. epoch:"<< num_circle << " , alpha_changes:" << alphaPairs_Change << endl;
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
                // cout << "search in alpha!=0; epoch:" << num_circle << " , alpha_changes:" << alphaPairs_Change<< endl;
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
        // cout << "epoch: " << num_circle <<  " label = " <<  _alphas << endl;
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
    // cout  << "ord " << ord << " Errori =  "  << Errori  << endl;
    // cout << " alpha = " << _alphas << endl;
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
        // double temp1 = _data.row(ord) * _data.row(JEJ[0]).transpose();
        // double temp2 = _data.row(ord) * _data.row(ord).transpose();
        // double temp3 = _data.row(JEJ[0]) * _data.row(JEJ[0]).transpose();
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
        // if (ord == 3)
        // {
        //     cout  << JEJ[0] << " "  << ord << " " << endl;
        // }
        updata_EK(ord);
        //update b1 and b2
        double b1 = _b - Errori - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp2 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp1;
        double b2 = _b - JEJ[1] - _label(ord,0)*(_alphas(ord,0) - alpha_i_old)*temp1 - _label(JEJ[0],0)*(_alphas(JEJ[0],0) - alpha_j_old)*temp3;
        // cout  << "ord = " << ord << " b = " << _b << endl;
        // cout << " errori " << Errori << endl;
        // cout << "alpha " << _alphas(JEJ[0],0)  << alpha_j_old << endl;
        // cout  << endl<< "what " << Kernel(1 ,1) << endl << endl;
        // cout << "b1 = " << b1 << " " << "b2 = " << b2 << endl;
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
        cout << "one is " << one.size() << " two is " << two.size() << endl;
    }
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(one.rows(),1);
    for (int i = 0; i < one.rows(); i++)
    {
        temp(i,0) = one(i,0)*two(i,0);
    }
    return temp;
    
}
double MYSVM::prediction(Eigen::MatrixXd test_datas , Eigen::MatrixXd test_labels)
{
    Eigen::MatrixXd valid_list = no_zero(_alphas);
    Eigen::MatrixXd temp;
    Eigen::MatrixXd predictionr_kernel = Eigen::MatrixXd(valid_list.size(),1);
    Eigen::MatrixXd label_alpha = Eigen::MatrixXd(valid_list.size(),1);
    Eigen::MatrixXd w = Eigen::MatrixXd(test_datas.row(0).size(),1);
    int num = 0;
    for (int i =0 ; i < valid_list.size() ; i++)
    {
        temp = _data.row(valid_list(i));
        _support_vectors.push_back(temp);//从后往前
        label_alpha(i,0) = _label(valid_list(i),0)*_alphas(valid_list(i),0);
    }
    // cout << predictionr_kernel.size() << endl;
    for ( int j = 0 ; j < test_datas.rows() ; j++)
    {
        // int j = 0;
        for (int k = 0 ; k < _support_vectors.size() ; k++)
        {
            auto temp2 = (_support_vectors[k] - test_datas.row(j) ) * (_support_vectors[k] - test_datas.row(j) ).transpose();
            predictionr_kernel(k,0) = std::exp(temp2.sum()/(-1*pow(1.3,2)));
            // w += label_alpha(k,0) * _support_vectors[k].transpose();
        }
        double type = ( (predictionr_kernel.transpose() * label_alpha).sum() + _b );
        // cout << (w.transpose() * test_datas.row(j).transpose()).size() << endl;
        // double type = ( (w.transpose() * test_datas.row(j).transpose()).sum() + _b );
        // if(flag == false)
        // {
        //     type = type - 0.731;
        // }
        if (test_labels(j,0)*type  > 0)
        {
            num++;
            // cout << "correct : " << type << endl;
            cout << "把第 " << j << " 个样本正确分类了。。。。应该是 " << test_labels(j,0) << " 分类样本是 type > 0  : " << type << endl;
        }
        else
        {
            cout << "把第 " << j << " 个样本错分了。。。。应该是 " << test_labels(j,0) << " 分类样本是 type < 0  : " << type << endl;
            cout << j << endl;
        }
    }
    // cout << num << endl;
    return double(num)/double(test_datas.rows());
}
void MYSVM::test(string filepath)
{
    Eigen::MatrixXd test_data_label = openData(filepath);
    Eigen::MatrixXd test_datas = test_data_label.block(0,0,test_data_label.rows() , test_data_label.cols()-1);
    Eigen::MatrixXd test_labels = test_data_label.block(0,test_data_label.cols()-1,test_data_label.rows() , 1);
    // test_datas = maxmin(test_datas);
    double rate = prediction(test_datas , test_labels);
    cout  << " 正确率为 ： " << rate << endl;
}
//matrix
double MYSVM::calculate_error(int ord)
{
    Eigen::MatrixXd temp = dot(_alphas,_label).transpose()*Kernel.col(ord);
    double Error_ord = temp.sum() + _b - _label(ord,0);
    // if (ord == 4)
    // {
    //     cout << "ord= "<< ord << endl;
    //     cout << "temp = " << temp.sum() + _b  << " b =  " << _b << " Error_ord =  " << Error_ord  << " " << endl;
    // }
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