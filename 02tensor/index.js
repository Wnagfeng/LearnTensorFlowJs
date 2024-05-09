import * as tf from "@tensorflow/tfjs";

// -------------------------------------------第一章(Hello, World!)--------------------------------------
// Create a tensor with a value of 1
const t0 = tf.tensor(1);
t0.print();
console.log(t0);

// Create a tensor with a value of [1, 2, 3]
const t1 = tf.tensor([1, 2, 3]);
t1.print();
console.log(t1);

//   Create a tensor with a value of [[1, 2], [3, 4]]
const t2 = tf.tensor([
  [1, 2],
  [3, 4],
]);
t2.print();
console.log(t2);

//  Create a tensor with a value of [[1, 2], [3, 4], [5, 6]]
const t3 = tf.tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);
t3.print();
console.log(t3);
//  tensor和机器学习有什么关系呢？
//  1. 机器学习的输入数据都是张量，张量可以是向量、矩阵、三维张量等。
//  2. 机器学习的输出数据也是张量，张量可以是向量、矩阵、标量等。
//  3. 机器学习的模型可以是神经网络、决策树、支持向量机等。
//  4. 机器学习的训练过程就是优化模型参数，使得模型的输出结果与真实值尽可能接近。

// -------------------------------------------第二章(for with tensor)--------------------------------------
// 传统循环
const input = [1, 2, 3, 4];
const w = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
];
const output = [0, 0, 0, 0];
for (let i = 0; i < w.length; i++) {
  for (let j = 0; j < input.length; j++) {
    output[i] += input[j] * w[i][j];
  }
}
console.log(output);

// 新款循环
tf.tensor(w).dot(tf.tensor(input)).print();
