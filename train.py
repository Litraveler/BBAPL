import conf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import util

# 设置GPU显存动态增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:  # 如果检测到GPU
    for gpu in gpus:  # 遍历每个GPU（多卡场景也适用）
        # 开启GPU显存的“动态增长”模式
        tf.config.experimental.set_memory_growth(gpu, True)

# Load and verify the dataset
# 【修改1】移除验证集加载相关代码
FOLDER_NPY = "TrainTestData/";
print("  Training samples...");
xt = np.load(FOLDER_NPY + "xt.npy", allow_pickle=True).item();
SEQUENCE_LENGTH = conf.DIMENTION;
INPUT_FEATURES = conf.FEATURES;

# Initialize the training generator (移除验证生成器)
descriptor = {};
descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH;
descriptor["INPUT_FEATURES"] = INPUT_FEATURES;

print("Initializing generators...");
print("  N (samples per user):        " + str(conf.N));
print("  K (sets per batch):          " + str(conf.K));
print("  Sample length (keystrokes):  " + str(SEQUENCE_LENGTH));
print("  Input features:              " + str(INPUT_FEATURES));

# 【修改2】只保留训练生成器初始化
print("  Initializing training generator...");
from DataGenerator import training_generator_save as dataGenerator
ti, tg, tc = dataGenerator.get_generator(descriptor, xt);

print("Initializing loss function...");
# 距离损失函数
from Loss import Distance_Similarity as loss;
l = loss.get_loss();  # 传递启动epoch参数

# # 【关键修改1】初始化损失时指定启动质心稳定损失的epoch（比如100）
# START_STAB_EPOCH = 0  # 自定义：从第100个epoch启动质心稳定损失
# l = Distance_Similarity.get_loss(start_stab_epoch=START_STAB_EPOCH);  # 传递启动epoch参数

from Models import SplitArchitecture as model;
descriptor = model.get_model(descriptor);
m = descriptor["model"];
m.summary();
# path = "./modelCos/final_model_weights"
# m.load_weights(path)
m.compile(
    optimizer=descriptor["optimizer"],
    loss = l  # 损失实例传入模型
);
# Initialize callbacks
class SaveModelFromEpochCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, save_start_epoch=10, **kwargs):
        super(SaveModelFromEpochCallback, self).__init__(filepath, **kwargs)
        self.save_start_epoch = save_start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.save_start_epoch:
            super(SaveModelFromEpochCallback, self).on_epoch_end(epoch, logs)

# 【关键修改2】重写UpdateLossEpochCallback：更新损失实例的epoch，而非模型
class UpdateLossEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_instance):
        super().__init__()
        self.loss_instance = loss_instance  # 接收损失实例

    def on_epoch_begin(self, epoch, logs=None):
        # 调用损失类的reset_epoch_cache方法，传入当前epoch（epoch从0开始，需+1匹配人类计数）
        current_epoch = epoch + 1
        self.loss_instance.reset_epoch_cache(current_epoch)
        print(f"\n[Callback] 重置损失缓存，当前epoch更新为：{current_epoch}")


# 自定义学习率调度
def lr_schedule(epoch):
    if epoch < 50:
        return 0.001
    elif epoch < 100:
        return 0.0005
    elif epoch < 150:
        return 0.0001
    elif epoch < 200:
        return 0.00005
    else:
        return 0.00001

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=2)
print("Initializing callbacks...");

# 【修改3】回调列表重构：传递损失实例，新增质心损失计算回调
callbacks = [
    UpdateLossEpochCallback(loss_instance=l),  # 传递损失实例，用于动态margin
    lr_callback
];

# 只保留训练生成器的回调
if tc != None:
    callbacks.append(tc);

# Train the model
print("Training model...");
history = descriptor["model"].fit(
    ti,
    epochs              = conf.EPOCHS,                    
    steps_per_epoch     = conf.TRAINING_STEPS,
    callbacks           = callbacks,
    verbose             = 2,
    # 新增：防止生成器输出形状不匹配的警告
    workers=0
);

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Core Train Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig("LOSS.png")
print("训练完成，开始保存模型权重...")

# 保存权重（TensorFlow 原生格式，永不报错）
model = descriptor["model"]
model.save_weights("modelDis/final_model_weights")  # 👈 这行就行

print("✅ 模型权重已保存到：modelDis/final_model_weights")
# 保存训练历史（仅核心损失）
np.save("training_history.npy", history.history['loss'])
print("训练完成！损失曲线已保存为LOSS.png，训练历史已保存为training_history.npy")