import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
window.onload = async function () {
  // 准备训练数据
  const heights = [150, 160, 170, 180, 190, 200];
  const weights = [40, 50, 60, 70, 80, 90];
  tfvis.render.scatterplot(
    { name: "身高体重训练数据！", tab: "Scatter Plot" },
    { values: heights.map((x, i) => ({ x, y: weights[i] })) },
    // { xLabel: "Input", yLabel: "Output" },一旦设置了x和y的别名 x和y的区间就不好使了 不知道为啥 2024年5月10日12点49分
    { xAxisDomain: [140, 210], yAxisDomain: [30, 100], zoomToFit: true }
  );
  //   将heights 进行归一化处理 150-200 映射到 0-1 之间 数字减去起始数字 除以区间
  const inputs = tf.tensor(heights).sub(150).div(50); //这里用了广播算法 tensor传入数组 .sub(150)会自动广播到每一个元素 减去起始数字 除以区间
  //   inputs.print();
  //   将weights 进行归一化处理 40-90 映射到 0-1 之间 数字减去起始数字 除以区间
  const outputs = tf.tensor(weights).sub(40).div(50);
  //   outputs.print();

  // 训练模型 进行预测 和反预测
  // create model and train it
  const model = tf.sequential(); //1.初始化模型
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true })); //2.添加全连接层(dense)
  //解析：units 神经元个数 inputShape 输入维度(最少是一维) useBias 是否使用偏置
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1),
  }); // 3.编译模型，设置均方误差和优化器
  // 解析：loss 损失函数，均方误差是回归问题常用的损失函数 ；optimizer 优化器，随机梯度下降SGD

  // 5.训练模型 batchSize 批次大小 epochs 训练轮数
  await model.fit(inputs, outputs, {
    batchSize: 4,
    epochs: 400,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]), // 6.显示训练过程
  });

  //    预测模型升高预测体重

  const prediction = model.predict(tf.tensor([165]).sub(150).div(50)); // 7.预测升高165cm的体重
  //   反归一化展示预测结果
  alert(
    "升高165cm的体重预测为：" + prediction.mul(50).add(40).dataSync()[0] + "kg"
  ); // 8.显示预测结果
};
