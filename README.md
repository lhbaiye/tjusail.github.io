# 结构化数据

---

搭建一个demo

1. 一个结构化数据的DNN模型预测
2. 一个结构化数据的传统模型预测
3. 添加一列，保证预测效果要变好（暂定ACC）
4. 分别对1和2进行攻击。

数据集和模型有推荐的嘛？

没有的话我就从kaggle上找几个流行的，模型找精确率最高的

done

---

## Dataset

### ****Iris Flower Dataset****

[Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?select=IRIS.csv)

数据集太小了，只有150个，但是很流行，而且只有四列，全是离散型数据

### ****Mobile Price Classification****

[Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

手机价格分类，流行，离散数据多

训练集2000个，预测集1000个，20列

### 糖尿病数据集

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

数据集由多个医学预测变量和一个目标变量 .预测变量包括患者的怀孕次数、BMI、胰岛素水平、年龄等

9列，一共有数据集900个

### ****健康保险交叉销售预测****

[Health Insurance Cross Sell Prediction 🏠 🏥](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction?select=train.csv)

以预测过去一年的投保人（客户）是否也会对公司提供的车辆保险感兴趣。

人口统计数据（性别、年龄、地区代码类型）、车辆（车龄、损坏）、保单（保费、采购渠道）等的信息。

---

暂时采用了 ****Mobile Price Classification数据集****

删除了`'px_width', 'px_height'` 两个属性，模型预测acc为0.787

增加这两个属性，模型精度为0.92

这两个属性应该是代表的屏幕像素，是连续值，可以稍微修改。符合实际逻辑

**接下来，修改一下代码结构，然后实现攻击方法。**

1. 模型1的攻击  测试集541/600 训练集：1118/1600
2. 模型2的攻击  测试集467/600  训练集：968/1600
3. 只对新增列的攻击  

- [ ]  代码修改，将攻击方法那里可拓展性强一点，用户可以选择攻击哪一列