# ========== 解决上层目录导入 conf.py ==========
import os
import tensorflow as tf
import numpy as np
import conf

# ======================================
# 所有自定义层 100% 强制命名，彻底修复
# ======================================
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name="kernel_W"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name="bias_b"
        )
        self.v = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="vector_v"
        )
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        scores = tf.matmul(scores, tf.expand_dims(self.v, -1))
        scores = tf.squeeze(scores, -1)
        att_weights = tf.nn.softmax(scores, axis=1)
        return tf.concat([inputs, tf.expand_dims(att_weights, -1)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, input_dim, dropout=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.input_dim = input_dim
        self.dropout = dropout

        # 强制命名所有内部层
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout,
            name="mha"
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1), name="ffn_dense1"),
            tf.keras.layers.Dense(input_dim, name="ffn_dense2"),
            tf.keras.layers.Dropout(dropout, name="ffn_dropout")
        ], name="ffn_stack")
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.dropout1 = tf.keras.layers.Dropout(dropout, name="dropout1")
        self.dropout2 = tf.keras.layers.Dropout(dropout, name="dropout2")

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "input_dim": self.input_dim,
            "dropout": self.dropout
        })
        return config

class FeatureAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform', trainable=True,
            name="w1"
        )
        self.b1 = self.add_weight(
            shape=(self.units,), initializer='zeros', trainable=True,
            name="b1"
        )
        self.W2 = self.add_weight(
            shape=(self.units, 1),
            initializer='glorot_uniform', trainable=True,
            name="w2"
        )
        self.b2 = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True,
            name="b2"
        )
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.tanh(tf.matmul(inputs, self.W1) + self.b1)
        scores = tf.matmul(scores, self.W2) + self.b2
        att_weights = tf.nn.softmax(scores, axis=1)
        return inputs * att_weights, att_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def get_sinusoidal_position_encoding(seq_len, feat_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(feat_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(feat_dim))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.expand_dims(tf.convert_to_tensor(angle_rads, tf.float32), 0)

def single_feature_encoder(input_tensor, seq_len, feature_dim, encoder_name):
    x = tf.keras.layers.Dense(
        feature_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_dense"
    )(input_tensor)
    bn = tf.keras.layers.BatchNormalization(name=f"{encoder_name}_bn")(x)

    ta1 = TemporalAttention(units=conf.MODEL_WIDTH, name=f"{encoder_name}_ta1")(bn)
    pos_emb = get_sinusoidal_position_encoding(seq_len, ta1.shape[-1])
    pos_encoded = ta1 + pos_emb

    num_heads = 4
    trans_encoder = TransformerEncoder(
        num_heads=num_heads,
        key_dim=ta1.shape[-1] // num_heads,
        ff_dim=conf.MODEL_WIDTH,
        input_dim=ta1.shape[-1],
        dropout=conf.MODEL_DROPOUT,
        name=f"{encoder_name}_transformer"
    )
    trans_out1 = trans_encoder(pos_encoded)
    trans_out2 = trans_encoder(trans_out1)
    trans_flatten = tf.keras.layers.Flatten(name=f"{encoder_name}_flatten")(trans_out2)
    trans_dense1 = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*2, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_td1"
    )(trans_flatten)
    trans_dense2 = tf.keras.layers.Dense(
        conf.MODEL_WIDTH, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_td2"
    )(trans_dense1)

    ta2 = TemporalAttention(units=conf.MODEL_WIDTH, name=f"{encoder_name}_ta2")(bn)
    conv1 = tf.keras.layers.Conv1D(
        2*conf.MODEL_FILTERS, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_conv1"
    )(ta2)
    conv2 = tf.keras.layers.Conv1D(
        conf.MODEL_FILTERS, 6, strides=2, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_conv2"
    )(conv1)
    conv_gap = tf.keras.layers.GlobalAveragePooling1D(name=f"{encoder_name}_gap")(conv2)

    concat = tf.keras.layers.Concatenate(name=f"{encoder_name}_concat")([trans_dense2, conv_gap])
    feat_vec1 = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*4, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_fv1"
    )(concat)
    feat_vec2 = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*2, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name=f"{encoder_name}_fv2"
    )(feat_vec1)
    return feat_vec2

def global_feature_encoder(input_tensor, seq_len, dropout_rate):
    expand_dim = conf.MODEL_WIDTH
    x = tf.keras.layers.Dense(
        expand_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name="global_dense"
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization(name="global_bn")(x)
    pos_emb = get_sinusoidal_position_encoding(seq_len, expand_dim)
    x = x + pos_emb

    num_heads = 4
    trans_encoder1 = TransformerEncoder(
        num_heads=num_heads,
        key_dim=expand_dim // num_heads,
        ff_dim=conf.MODEL_WIDTH*2,
        input_dim=expand_dim,
        dropout=dropout_rate,
        name="global_transformer"
    )
    x = trans_encoder1(x)
    x = trans_encoder1(x)
    x = tf.keras.layers.Flatten(name="global_flatten")(x)
    x = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*16, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name="global_td1"
    )(x)
    x = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*8, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name="global_td2"
    )(x)
    return tf.keras.layers.Dense(
        conf.MODEL_WIDTH*8, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name="global_out"
    )(x)

def get_model(descriptor):
    print("Model configuration:")
    print("  WIDTH:   " + str(conf.MODEL_WIDTH))
    print("  FILTERS: " + str(conf.MODEL_FILTERS))
    print("  DROPOUT: " + str(conf.MODEL_DROPOUT))
    print("Compiling model...")

    retval = descriptor
    retval["name"] = "StrokePL_FeatureAttention"
    retval["optimizer"] = tf.keras.optimizers.Adam(0.001)
    retval["normalized"] = False
    retval["epochs"] = 600
    retval["threshold"] = 0.86
    seq_len = descriptor["SEQUENCE_LENGTH"]
    num_features = descriptor["INPUT_FEATURES"]
    input_layer = tf.keras.layers.Input(shape=(seq_len, num_features), name="model_input")

    feature_splits = tf.split(input_layer, num_features, axis=-1)
    feature_vectors = [
        single_feature_encoder(feat, seq_len, conf.RISE_WIDTH, f"feat_{i}")
        for i, feat in enumerate(feature_splits)
    ]
    stacked_feats = tf.stack(feature_vectors, axis=1, name="stack_feats")

    weighted_feats, att_weights = FeatureAttention(
        units=conf.MODEL_WIDTH, name="feature_att"
    )(stacked_feats)

    attended_feats = tf.keras.layers.MultiHeadAttention(
        num_heads=8, key_dim=64, dropout=conf.MODEL_DROPOUT,
        name="final_mha"
    )(weighted_feats, weighted_feats)

    attended_feats = tf.keras.layers.Add(name="add_feats")([attended_feats, weighted_feats])
    attended_feats = tf.keras.layers.LayerNormalization(name="ln_feats")(attended_feats)

    x = tf.keras.layers.Flatten(name="final_flatten")(attended_feats)
    x = tf.keras.layers.Dense(
        conf.MODEL_WIDTH*32, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
        name="final_dense1"
    )(x)
    final_out = tf.keras.layers.Dense(conf.MODEL_WIDTH*16, name="final_out")(x)

    # global_feat = global_feature_encoder(input_layer, seq_len, conf.MODEL_DROPOUT)

    # combined = tf.keras.layers.Concatenate(name="combine_feat")([final_out, global_feat])
    combined = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=-1), name="l2_norm"
    )(final_out)

    retval["model"] = tf.keras.Model(
        inputs=input_layer, outputs=combined, name="StrokePL"
    )
    retval["feature_attention_weights"] = att_weights
    return retval

# ==================== MAIN 测试：一键验证保存 ====================
if __name__ == '__main__':
    # 构建模型
    test_desc = {
        "SEQUENCE_LENGTH": 150,
        "INPUT_FEATURES": 17
    }
    model_desc = get_model(test_desc)
    model = model_desc["model"]

    # 必须前向传播初始化权重
    test_input = tf.random.normal((1, 150, 17))
    output = model.predict(test_input, verbose=0)
    print(f"测试输出形状: {output.shape}")

    # 保存（彻底修复）
    os.makedirs("model2", exist_ok=True)
    save_path = "./model2/final_model_weights"

    # 关键：使用 TensorFlow 原生格式保存
    model.save_weights(save_path, save_format="tf")
    print(f"\n✅ 保存成功！路径：{save_path}")

    # 测试加载
    new_model = get_model(test_desc)["model"]
    new_model.load_weights(save_path)
    print("✅ 加载成功！模型完全正常！")