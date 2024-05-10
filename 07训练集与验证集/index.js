import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getIrisData, IRIS_CLASSES } from "./data.js";

window.onload = async () => {
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);
  /*  
  xTrain:是训练集的特征数据，
  yTrain:是训练集的标签数据，
  xTest:是测试集的特征数据，
  yTest:是测试集的标签数据。
  */
  /*
 Iris：数据是一个有四个特征数据集 ，想要解决这个问题 使用多层来解决概问题
 这一章我们将实现 一个带有多层神经网络的模型，来解决鸢尾花分类问题。
 之前我们解决二分类问题的时候我们使用的是一个神经网络使用一个激活函数是sigmoid函数，
 这会不行了我们有三类 怎么训练呢？请看 Code
  */
  /*
 1.初始化网络模型
 2.为神经网络添加两层
 3.设计神经元个数 inputShape 激活函数 
  */
  //  1.初始化网络模型
  const model = tf.sequential();

  //  2.为神经网络添加两层
  model.add(
    tf.layers.dense({
      inputShape: [xTrain.shape[1]], //这是输入层的神经元个数shape保存的就是个数
      units: 16, //第一层神经元个数
      activation: "sigmoid", //暂时用sigmoid函数
    })
  );
  //   第二层 (多分类神经网络的核心代码)
  model.add(
    tf.layers.dense({
      inputShape: [xTrain.shape[1]], //这是输入层的神经元个数shape保存的就是个数
      units: 3, //第二层神经元个数：由输出集个数决定  这样才能输出三个概率
      activation: "softmax", //最后一层用softmax函数
    })
  );
  //   训练模型
  // 设置交叉熵损失函数 和准确度度量

  model.compile({
    optimizer: tf.train.adam(0.01), //使用adam优化器
    loss: "categoricalCrossentropy", //使用交叉熵损失函数
    metrics: ["accuracy"], //使用准确度度量
  });
  await model.fit(xTrain, yTrain, {
    epochs: 100, //训练次数
    validationData: [xTest, yTest], //丢入测试集数据
    callbacks: tfvis.show.fitCallbacks({ name: "训练效果展示" }, [
      "loss",
      "val_loss",
      "acc",
      "val_acc",
    ]), //训练过程可视化
  });
  //   评估模型
  //   花萼长度：
  //   花萼宽度：
  //   花瓣长度：
  //   花瓣宽度：

  const result = model.predict(tf.tensor2d([[5.1, 3.5, 1.4, 0.2]]));//山鸢尾花的特征数据
  const result1= model.predict(tf.tensor2d([[6.3, 3.3, 4.7, 1.6]]));//变色鸢尾花的特征数据
  const result2= model.predict(tf.tensor2d([[4.6, 3.1, 1.5, 0.2]]));//维吉尼亚鸢尾花的特征数据
  const predictedClass = result1.argMax(1).dataSync()[0];
  alert(`预测结果：${IRIS_CLASSES[predictedClass]}`); //预测结果：setosa
};
