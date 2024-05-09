import * as tfvis from "@tensorflow/tfjs-vis";
window.onload = function () {
  const xs = [1, 2, 3, 4, 5]; // input data
  const ys = [1, 3, 2, 5, 4]; // output data
  tfvis.render.scatterplot(
    { name: "线性回归样本集", tab: "Scatter Plot" },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xLabel: "Input", yLabel: "Output" },
    { xAxisDomain: [0, 8], yAxisDomain: [0,8] }
  );
};
