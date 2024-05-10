import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "./data.js";

window.onload = async () => {
  const data = getData(400);
  console.log(data);
  //   渲染 模型
  tfvis.render.scatterplot(
    { name: "XOR训练数据", tab: "Scatter Plot" },
    {
      values: [
        data.filter((d) => d.label === 1),
        data.filter((d) => d.label === 0),
        // filter 函数过滤掉label为0的数据，label为1的数据
      ],
    },
    { zoomToFit: true }
  );
  //   创建模型

  const model = tf.sequential();
  //   创建隐藏层
  model.add(tf.layers.dense({ inputShape: [2], units: 4, activation: "relu" }));
  //   输出层
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  /*
  为什么需要多层神经网络？
  1. 解决非线性问题
  2. 增加模型复杂度，提高模型拟合能力
  3. 增加模型非线性，提高模型鲁棒性
  4. 增加模型参数量，提高模型泛化能力
   */
  //   训练多层神经网络模型
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.01),
  });

  const inputs = tf.tensor(data.map((d) => [d.x, d.y]));
  const labels = tf.tensor(data.map((d) => d.label));

  await model.fit(inputs, labels, {
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks({ name: "XOR训练过程" }, ["loss"]),
  });

  //    预测结果
  const predictData = [
    [-2, 2],
  ];
  const predictions = model.predict(tf.tensor(predictData));
  alert(`预测结果：${predictions.dataSync()}`);
};
