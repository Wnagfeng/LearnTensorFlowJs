import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";
import { getData } from "./data";

window.onload = async () => {
  const data = getData(500); //获取数据
  //   渲染 模型
  tfvis.render.scatterplot(
    { name: "逻辑回归训练数据", tab: "Scatter Plot" },
    {
      values: [
        data.filter((d) => d.label === 0),
        data.filter((d) => d.label === 1),
        // filter 函数过滤掉label为0的数据，label为1的数据
      ],
    },
    { zoomToFit: true }
  );
  //   定义模型结构： 带有激活函数的单个神经元
  //   我们为了输出0-1之间的概率 所以需要使用激活函数

  const model = tf.sequential();

  model.add(
    tf.layers.dense({ inputShape: [2], units: 1, activation: "sigmoid" })
  );
  //   sigmoid 函数将输出值压缩到0-1之间
  //   输入层有2个神经元，输出层有1个神经元，使用sigmoid激活函数
  //   在逻辑回归中 我们需要使用 对数损失函数 而不是 普通的 均方误差函数 看笔记
  model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.01) });
  //   编译模型，使用对数损失函数和adam优化器
  //   adam的作用是 自动调整学习率，使得模型在训练过程中更加稳定
  const inputs = tf.tensor(data.map((d) => [d.x, d.y])); //这是输入数据
  const labels = tf.tensor(data.map((d) => d.label)); //这是标签数据
  //   训练模型
  await model.fit(inputs, labels, {
    epochs: 500,
    batchSize: 500,
    callbacks: tfvis.show.fitCallbacks(
      { name: "训练过程", tab: "Training" },
      ["loss"],
      {
        yAxis: "loss",
      }
    ),
  });
  // 获取输入框和预测按钮的引用
  const xInput = document.getElementById("x-input");
  const yInput = document.getElementById("y-input");
  const predictButton = document.getElementById("predict");
  predictButton.onclick = () => {
    const xValue = parseFloat(xInput.value);
    const yValue = parseFloat(yInput.value);

    // 对用户输入进行验证
    if (isNaN(xValue) || isNaN(yValue)) {
      alert("请输入有效的数字！");
      return;
    }

    // 使用模型进行预测
    const prediction = model.predict(tf.tensor([[xValue, yValue]]));

    // 获取预测结果
    const result = prediction.dataSync()[0];

    // 显示预测结果
    alert(
      `您的输入坐标为：(${xValue},${yValue}), 预测结果为：${result.toFixed(2)}`
    );
  };
};
