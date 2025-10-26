<p style="margin-top: 0px; font-family: SimSun; font-size: 16px;"><a href="https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model">https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model</a></p>

<p style="text-align: unset; margin-top: 0px; font-family: SimSun; font-size: 16px;"><br /></p>

# <p style="margin-top: 0px;">step分隔符对应关系</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">openPangu中<code>\n\n</code>在tokenize后仍为其本身</p>

```python
text = "so yes, 1/3 = 4/12.\n\nAnd 5/12 is already in twelfths.\n\nSo, P(C) = 12/12 - 4/12 - 5/12\n\nNow,"
tokens = tokenizer.tokenize(text)
print(tokens)

# ['so', '▁yes', '<0x2C>', '▁1', '/3', '▁=', '▁4', '/12', '.\n\n', 'And', '▁5', '/12', '▁is', '▁already', '▁in', '▁tw', 'elf', 'ths', '.\n\n', 'So', '<0x2C>', '▁P', '(C', '<0x29>', '▁=', '▁12', '/12', '▁-', '▁4', '/12', '▁-', '▁5', '/12', '\n\n', 'Now', '<0x2C>']
```

# <p style="margin-top: 0px;">token特性</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">将text转换为id时若add_special_tokens=True，会在句首增加<code><s></code>，即bos_token</p>

# <p style="margin-top: 0px;">Chat Template</p>

```python
sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
problem = "This is a dummy problem."
messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": problem}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# [unused9]系统：Please reason step by step, and put your final answer within \boxed{}.[unused10][unused9]用户：This is a dummy problem.[unused10][unused9]助手：

```

# <p style="margin-top: 0px;">Thinking模式切换</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">openPangu是没有<code><think></code>与<code></think></code> token的</p>

* 慢思考模式：prompt（无任何设置）

* 快思考模式：prompt + " /no_think"

* 自动切换快慢思考模式：prompt + " /auto_think"

# <p style="margin-top: 0px;">CoT拆分</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">thinking content：[unused16] ...... [unused17]</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">content：[unused17] ...... [unused10]</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">其中：</p>

* [unused16]: 类似&lt;think> token

* [unused17]: 类似&lt;/think> token

* [unused10]: eos_token_id

🎯

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">发现[unused 17]并不能很好的结束thinking，后续依然会思考，并输出多个\n\n</p>

# <p style="margin-top: 0px;">Tokenizer特性</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">默认情况下（未明确指定add_special_tokens=False），使用tokenizer将文本直接转换为ids时均会在开头增加&lt;s>（id为1）。但若先将文本转换为token，再转换为id，则不加&lt;s>。</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">默认情况下，模型输出时，最后一个token为[unused10]（id为45892）（qwen系列也会输出&lt; | end_of_sentence | >）</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;"><br /></p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">模型输出的第一个token总是[unused16]。而qwen系列模型的&lt;think>是包含在提示词中的，并非由模型生成。</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;"><br /></p>

```python

# 当return_tensors="pt"时，prompt是否加[]，input_ids都为二维。当没有设置return_tensors="pt"时，prompt不加[]是一维，加[]是两维
# pangu与qwen都适用
input_ids = tokenizer([prompt], return_tensors="pt").input_ids
```

# <p style="margin-top: 0px;">Model Details</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">model</p>

```scss
PanguEmbeddedForCausalLM(
  (model): PanguEmbeddedModel(
    (embed_tokens): Embedding(153376, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-33): 34 x PanguEmbeddedDecoderLayer(
        (self_attn): PanguEmbeddedAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=True)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=True)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=True)
        )
        (mlp): PanguEmbeddedMLP(
          (gate_proj): Linear(in_features=4096, out_features=12800, bias=False)
          (up_proj): Linear(in_features=4096, out_features=12800, bias=False)
          (down_proj): Linear(in_features=12800, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): PanguEmbeddedRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): PanguEmbeddedRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): PanguEmbeddedRMSNorm((4096,), eps=1e-05)
    (rotary_emb): PanguEmbeddedRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=153376, bias=False)
)
```

<p style="text-align: unset; margin-top: 0px; font-family: SimSun; font-size: 16px;">model.config</p>

```java
PanguEmbeddedConfig {
  "architectures": [
    "PanguEmbeddedForCausalLM"
  ],
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_openpangu_dense.PanguEmbeddedConfig",
    "AutoModel": "modeling_openpangu_dense.PanguEmbeddedModel",
    "AutoModelForCausalLM": "modeling_openpangu_dense.PanguEmbeddedForCausalLM"
  },
  "bias": true,
  "bos_token_id": 1,
  "eos_token_id": 45892,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12800,
  "max_position_embeddings": 32768,
  "model_type": "PanguEmbedded",
  "num_attention_heads": 32,
  "num_hidden_layers": 34,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-05,
  "rope_theta": 16000000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.2",
  "use_cache": true,
  "vocab_size": 153376
}
```

# <p style="margin-top: 0px;">Layer Analysis</p>

## <p style="margin-top: 0px;">Version 0</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">选择layer 27（从1开始第27层，代码中输入layer_id=27）</p>

```yaml
🎯 各层 PCA+Ridge 测试集指标：
Layer  1: R² = 0.0591 | Spearman ρ = 0.2656
Layer  2: R² = 0.0725 | Spearman ρ = 0.2927
Layer  3: R² = 0.0742 | Spearman ρ = 0.2939
Layer  4: R² = 0.0676 | Spearman ρ = 0.2798
Layer  5: R² = 0.0663 | Spearman ρ = 0.2781
Layer  6: R² = 0.0666 | Spearman ρ = 0.2769
Layer  7: R² = 0.0671 | Spearman ρ = 0.2775
Layer  8: R² = 0.0658 | Spearman ρ = 0.2762
Layer  9: R² = 0.0661 | Spearman ρ = 0.2766
Layer 10: R² = 0.0662 | Spearman ρ = 0.2754
Layer 11: R² = 0.0681 | Spearman ρ = 0.2774
Layer 12: R² = 0.0686 | Spearman ρ = 0.2776
Layer 13: R² = 0.0714 | Spearman ρ = 0.2819
Layer 14: R² = 0.0719 | Spearman ρ = 0.2844
Layer 15: R² = 0.0710 | Spearman ρ = 0.2837
Layer 16: R² = 0.0742 | Spearman ρ = 0.2912
Layer 17: R² = 0.0738 | Spearman ρ = 0.2896
Layer 18: R² = 0.0733 | Spearman ρ = 0.2890
Layer 19: R² = 0.0715 | Spearman ρ = 0.2871
Layer 20: R² = 0.0720 | Spearman ρ = 0.2873
Layer 21: R² = 0.0721 | Spearman ρ = 0.2877
Layer 22: R² = 0.0736 | Spearman ρ = 0.2890
Layer 23: R² = 0.0739 | Spearman ρ = 0.2906
Layer 24: R² = 0.0770 | Spearman ρ = 0.2960
Layer 25: R² = 0.0776 | Spearman ρ = 0.2981
Layer 26: R² = 0.0790 | Spearman ρ = 0.3003
Layer 27: R² = 0.0796 | Spearman ρ = 0.3013
Layer 28: R² = 0.0787 | Spearman ρ = 0.2997
Layer 29: R² = 0.0777 | Spearman ρ = 0.2977
Layer 30: R² = 0.0782 | Spearman ρ = 0.2998
Layer 31: R² = 0.0770 | Spearman ρ = 0.2973
Layer 32: R² = 0.0796 | Spearman ρ = 0.2998
Layer 33: R² = 0.0792 | Spearman ρ = 0.2984
Layer 34: R² = 0.0668 | Spearman ρ = 0.2771
```

## <p style="margin-top: 0px;">Version 1</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">逻辑优化后</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">选择Layer 33</p>

```yaml
Layer  1: R² = 0.2476 | Spearman ρ = 0.5375
Layer  2: R² = 0.2864 | Spearman ρ = 0.5727
Layer  3: R² = 0.2990 | Spearman ρ = 0.5851
Layer  4: R² = 0.3006 | Spearman ρ = 0.5872
Layer  5: R² = 0.3135 | Spearman ρ = 0.5958
Layer  6: R² = 0.3214 | Spearman ρ = 0.6033
Layer  7: R² = 0.3272 | Spearman ρ = 0.6101
Layer  8: R² = 0.3277 | Spearman ρ = 0.6106
Layer  9: R² = 0.3370 | Spearman ρ = 0.6180
Layer 10: R² = 0.3424 | Spearman ρ = 0.6223
Layer 11: R² = 0.3466 | Spearman ρ = 0.6259
Layer 12: R² = 0.3483 | Spearman ρ = 0.6266
Layer 13: R² = 0.3523 | Spearman ρ = 0.6301
Layer 14: R² = 0.3576 | Spearman ρ = 0.6342
Layer 15: R² = 0.3661 | Spearman ρ = 0.6422
Layer 16: R² = 0.3722 | Spearman ρ = 0.6461
Layer 17: R² = 0.3807 | Spearman ρ = 0.6527
Layer 18: R² = 0.3902 | Spearman ρ = 0.6599
Layer 19: R² = 0.3973 | Spearman ρ = 0.6655
Layer 20: R² = 0.4059 | Spearman ρ = 0.6718
Layer 21: R² = 0.4078 | Spearman ρ = 0.6735
Layer 22: R² = 0.4154 | Spearman ρ = 0.6796
Layer 23: R² = 0.4252 | Spearman ρ = 0.6859
Layer 24: R² = 0.4274 | Spearman ρ = 0.6883
Layer 25: R² = 0.4303 | Spearman ρ = 0.6904
Layer 26: R² = 0.4294 | Spearman ρ = 0.6899
Layer 27: R² = 0.4274 | Spearman ρ = 0.6877
Layer 28: R² = 0.4273 | Spearman ρ = 0.6883
Layer 29: R² = 0.4272 | Spearman ρ = 0.6879
Layer 30: R² = 0.4268 | Spearman ρ = 0.6871
Layer 31: R² = 0.4269 | Spearman ρ = 0.6885
Layer 32: R² = 0.4236 | Spearman ρ = 0.6859
Layer 33: R² = 0.4332 | Spearman ρ = 0.6925
Layer 34: R² = 0.3637 | Spearman ρ = 0.6384
```

# <p style="margin-top: 0px;">Hidden Analysis</p>

## <p style="margin-top: 0px;">Version 0</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">threshold: 0.75</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">温和距离：alpha_mean_S = 1.14</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">激进距离：alpha_all_S = 5.54</p>

```python
# hidden analysis report
{
  "layer_id": 27,
  "dim": 4096,
  "n_hit": 47345,
  "n_non": 55509,
  "S_l2": 18.099870681762695,
  "hit_to_non_min": 65.78202056884766,
  "hit_to_non_max": 184.5203857421875,
  "non_to_hit_min": 64.67111206054688,
  "non_to_hit_max": 180.79078674316406,
  "jsonl_path": "/home/ma-user/work/dataset/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.merged.jsonl",
  "hidden_dir": "/home/ma-user/work/dataset/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/",
  "threshold": 0.75,
  "center_block_size": 65536,
  "device": "npu",
  "confidence_q25": 0.7507111579179764,
  "confidence_q75": 0.9260406494140625,
  "confidence_N": 125010,
  "w_norm": 2.055134690625292,
  "S_norm": 18.09987190130844,
  "b": 8.807880123351854,
  "threshold_t": -8.807880123351854,
  "max_w_dot_h": 143.5134631382569,
  "margin_needed": 74.11745028506603,
  "d_all_w": 74.11745028506603,
  "d_all_S": 100.37578836948285,
  "alpha_all_S": 5.545662915007766,
  "d_mean_w": 15.25405413371759,
  "d_mean_S": 20.658262036992678,
  "alpha_mean_S": 1.141348521670927,
  "w_dot_uS": 1.517510803509848
}
```

## <p style="margin-top: 0px;">Version 1</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">threshold: 0.74</p>

<p style="text-align: left; margin-top: 0px; font-family: SimSun; font-size: 16px;">温和距离：alpha_mean_S = 0.99</p>

<p style="text-align: start; margin-top: 0px; font-family: SimSun; font-size: 16px;">激进距离：alpha_all_S = 3.97</p>

```json
{
  "layer_id": 33,
  "dim": 4096,
  "n_hit": 50072,
  "n_non": 65542,
  "S_l2": 34.81904983520508,
  "hit_to_non_min": 111.92693328857422,
  "hit_to_non_max": 365.236572265625,
  "non_to_hit_min": 114.32675170898438,
  "non_to_hit_max": 353.8177795410156,
  "jsonl_path": "/home/ma-user/work/dataset/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.jsonl",
  "hidden_dir": "/home/ma-user/work/dataset/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/",
  "threshold": 0.74,
  "center_block_size": 65536,
  "device": "npu",
  "confidence_q25": 0.7398352473974228,
  "confidence_q75": 0.9247501939535141,
  "confidence_N": 126722,
  "w_norm": 1.5205217857976203,
  "S_norm": 34.819037596402964,
  "b": 9.273947276990937,
  "threshold_t": -9.273947276990937,
  "max_w_dot_h": 141.49369775399822,
  "margin_needed": 99.15520214121877,
  "d_all_w": 99.15520214121877,
  "d_all_S": 138.09312030926014,
  "alpha_all_S": 3.9660234699744286,
  "d_mean_w": 24.698956613311317,
  "d_mean_S": 34.39815474590563,
  "alpha_mean_S": 0.9879122778930052,
  "w_dot_uS": 1.0917824486347056
}
```

# <p style="margin-top: 0px;">TODO List</p>

<p style="margin-top: 0px; font-family: SimSun; font-size: 16px;">当前提取thinking contents时包含了初始的[unused16]，是否需排除？</p>

<p style="margin-top: 0px; font-family: SimSun; font-size: 16px;">当前低置信词表复用qwen，是否需重新构建？</p>

<p style="margin-top: 0px; font-family: SimSun; font-size: 16px;">从gen_logprobs中取confidence时应该跳过\n\n，即start=end+1</p>

<p style="margin-top: 0px; font-family: SimSun; font-size: 16px;">使得所有hidden state与conf中的step数量都刚好相差1可能是不现实的</p>

```python
# 这里我们要算conf，就只能算\n\n和------\n\n这两个token的conf，没有方式能够得到------的conf，因为其与\n\n是一起生成的

# test
b = "so:\n\nAlign:\n\n  8855\n\n+33649\n\n------\n\nStart from right:\n\n5+9=14"
tokenizer.tokenize(b)

# ['so',
#  ':\n\n',
#  'Align',
#  ':\n\n',
#  '▁▁',
#  '88',
#  '55',
#  '\n\n',
#  '<0x2B>',
#  '33',
#  '649',
#  '\n\n',
#  '------\n\n',
#  'Start',
#  '▁from',
#  '▁right',
#  ':\n\n',
#  '<0x35>',
#  '<0x2B>',
#  '<0x39>',
#  '<0x3D>',
#  '14']
```

<p style="text-align: unset; margin-top: 0px; font-family: SimSun; font-size: 16px;"><br /></p>
