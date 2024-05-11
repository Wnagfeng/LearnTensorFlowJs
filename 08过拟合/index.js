import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "./data";
window.onload = async () => {
  const data = getData(500, 10);
  /*
  data 是带有噪点的数据
  什么是噪点？
  噪点是指数据中存在一些异常值，这些异常值对模型的训练有着不利的影响。 
   */
  tfvis.render.scatterplot(
    { name: "过拟合训练数据", tab: "Scatter Plot" },
    {
      values: [
        data.filter((d) => d.label === 1),
        data.filter((d) => d.label === 0),
        // filter 函数过滤掉label为0的数据，label为1的数据
      ],
    },
    { zoomToFit: true }
  );

  const model = tf.sequential();
  // 解决方案：
  // 1.权重衰减（L2正则化）
  model.add(
    tf.layers.dense({
      inputShape: [2],
      units: 10,
      activation: "tanh",
      // kernelRegularizer: tf.regularizers.l2({ l2: 1.4 }), // 1.权重衰减（L2正则化）
    })
  );
  model.add(tf.layers.dropout({ rate: 0.9 }));// 2.防止过拟合，随机丢弃一些神经元(这个一定要加在复杂模型后面)
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));


  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.01),
  });

  const inputs = tf.tensor(data.map((d) => [d.x, d.y]));
  const labels = tf.tensor(data.map((d) => d.label));

  await model.fit(inputs, labels, {
    epochs: 200,
    validationSplit: 0.2, // 验证集比例（为了看见验证集合我们从训练集中分出20%作为验证集）
    callbacks: tfvis.show.fitCallbacks({ name: "过拟合训练效果展示！" }, [
      "loss",
      "val_loss",
    ]),
  });
};
