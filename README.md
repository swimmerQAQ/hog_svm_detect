# 支持向量机 svm：基于 Eigen 矩阵库和 Opencv
## 概述
这是一个**hog特征提取库**和**svm(核为rbf)的分类器**的两个库，我只测试了一些不多的数字分类，对常规的图像没有测试过效果，手写数字到还可以（ac_rate>=0.972），后续更新库中的详细注释，希望可以给出一些启发和总结。
>>我认为还**有待完善的问题**
>
> hog特征提取里面的**滤波**，我感觉是导致最后干扰较大，我测试的数据集效果确实有点问题。。。心累。。。（干扰较大）
>
> 此外我觉得应该还有更多更佳的**特征提取**的方案。。。
>> 总结
>
> 使用Eigen库差点没把我送走，后续再总结吧。
>
> 对我来说完全自己独立思考和实现难度还是太大，但是独立尝试帮助还是挺大的。
>
> 感觉机器学习有很多东西可以挖掘的，芜湖起飞！
>
> ~~感觉github真不戳～~~
>

## 非常感谢https://github.com/Jack-Cherish 的算法指导
