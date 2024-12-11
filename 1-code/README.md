这里是中国科学院大学本科部2023-2024学年秋季学期机器学习课鸡泣学习小组的大作业，组长杨镕争，组员陈翼飞，王攀宇，侯汝垚，张钊珲

本实验的实验环境为python3.9,用到了TensorFlow、Numpy、pandas、sklearn、lightgbm等库，注意TensorFlow的版本为2.12，更新版本可能会造成变量名冲突

项目共完成了三种模型，分别为梯度提升决策树（LGBM）、线性回归模型（Linear）、神经网络（Neural networks），模型堆叠（StackingRegressor），分别对应压缩包中的lgbm.py，linear.py，net.py，combine.py

其中线性模型中实现了基础的正向反向传播的线性模型到加入了正则化的最小二乘法线性模型，需要运行不同模型只需要在261行开始的部分注释掉其他模型即可；
堆叠模型将线性回归模型和神经网络堆叠后，使用线性模型作为元模型进行堆叠

注意在运行的时候将文件改名，否则会遇到文件名和库函数名冲突的情况

此外尝试了五折数据划分，但遗憾的是不同模型的得分都有下降，猜测可能是由于特征是基于时间的序列，并不适合打乱后训练，由于能力受限，也没有找到更合适的数据处理的方式