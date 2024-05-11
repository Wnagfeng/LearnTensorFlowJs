import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "../06XOR多模型/data";
window.onload = async () => {
  const data = getData(500);
  /*
  只有用简单模型处理复杂数据 才能展现欠拟合现象
  欠拟合就是训练损失始终在天上飘着 下不来
   */
  tfvis.render.scatterplot(
    { name: "过拟合训练数据", tab: "Scatter Plot" },
    {
      values: [
        data.filter((d) => d.label === 1),
        data.filter((d) => d.label === 0),
      ],
    },
    { zoomToFit: true }
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [2], units: 1, activation: "sigmoid" })
  );
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.01),
  });

  const inputs = tf.tensor(data.map((d) => [d.x, d.y]));
  const labels = tf.tensor(data.map((d) => d.label));

  await model.fit(inputs, labels, {
    epochs: 200,
    validationSplit: 0.2, // 验证集比例（为了看见验证集合我们从训练集中分出20%作为验证集）
    callbacks: tfvis.show.fitCallbacks({ name: "欠拟合训练效果展示！" }, [
      "loss",
      "val_loss",
    ]),
  });
};
