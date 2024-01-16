# ETTh1的时间序列预测

`etth1.sh`为调用主脚本，通过调节model_name可更换模型种类，调节seq_len可更改预测的序列长度。不同的序列长度会重新训练。此外，还可以调节其他相关超参数进行性能的调优。

`model`文件夹下存放的是包含LSTM、Transformers、our_model在内的模型架构。

其余的脚本皆为辅助脚本，主要为Transformers模型的子模块实现，参考实现来自[Autoformer](https://github.com/thuml/Autoformer)