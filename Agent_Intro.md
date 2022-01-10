## DQNAgent

### DQNAgent():

\__init__:

step(): 环境调用，e-greedy，返回选择的action

eval_step(): 环境调用，greedy，返回选择的action

feed(): 外部调用，存储(s,a,s’,r)四元组

modify_state(): 内部调用，将环境的state变为我们需要的（有待优化）

train(): 外部调用，对self.memory进行采样、更新q_net和target_net

 

### Memory():

\__init__:

save(): 被feed调用，存储四元组

sample(): 被train调用，采样，返回样本index及样本

update_TDerror(): 更新被采样样本的TDerror，随之改变prob和lr

 

### Network():

\__init__(): 定义网络

forward(): 内部调用，被Qvalue调用，正向传播

Qvalue(): 返回输入state对应的Qvalues（共5个）

update(): 反向传播，更新参数