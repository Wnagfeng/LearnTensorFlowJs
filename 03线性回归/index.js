import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
window.onload = async function () {
  const xs = [1, 2, 3, 4]; // input data
  const ys = [1, 3, 5, 7]; // output data
  // render scatter plot of input vs output data
  tfvis.render.scatterplot(
    { name: "线性回归样本集", tab: "Scatter Plot" },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xLabel: "Input", yLabel: "Output" },
    { xAxisDomain: [0, 8], yAxisDomain: [0, 8] }
  );
  // create model and train it
  const model = tf.sequential(); //1.初始化模型
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true })); //2.添加全连接层(dense)
  //解析：units 神经元个数 inputShape 输入维度(最少是一维) useBias 是否使用偏置
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1),
  }); // 3.编译模型，设置均方误差和优化器
  // 解析：loss 损失函数，均方误差是回归问题常用的损失函数 ；optimizer 优化器，随机梯度下降SGD

  // 4.将数据转换为tensor形式
  const inputsTensor = tf.tensor(xs);
  const lebelsTensor = tf.tensor(ys);

  // 5.训练模型 batchSize 批次大小 epochs 训练轮数
  await model.fit(inputsTensor, lebelsTensor, {
    batchSize: 5,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]), // 6.显示训练过程
  });

  // 验证模型
  const output = model.predict(tf.tensor([5])); // 7.预测输出
  console.log(output.dataSync()[0].toFixed(2)); // 打印输出结果
  alert(`输入5，预测输出为：${output.dataSync()[0].toFixed(2)}`); // 弹窗显示预测输出
};
