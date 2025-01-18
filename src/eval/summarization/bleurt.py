import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

# 加载模型和分词器
config = BleurtConfig.from_pretrained('/Users/jonnyw/Documents/nju/graduation_proj/mydataset/code/localmodel/BLEURT-20-D12')
model = BleurtForSequenceClassification.from_pretrained('/Users/jonnyw/Documents/nju/graduation_proj/mydataset/code/localmodel/BLEURT-20-D12')
tokenizer = BleurtTokenizer.from_pretrained('/Users/jonnyw/Documents/nju/graduation_proj/mydataset/code/localmodel/BLEURT-20-D12')

# 三组参考语句和待评估语句
references = [
    """Foreign institutions expressed confidence in China's economic growth and A-share market value during meetings with Shanghai and Shenzhen Stock Exchanges, emphasizing long-term investment in technology, manufacturing, and consumer electronics. Goldman Sachs predicts a 13% CSI 300 Index rise in 2025, while relaxed policies and strong inflows are expected to boost valuations and investor confidence.""",
    """Foreign institutions expressed confidence in China's economic growth and A-share market value during meetings with Shanghai and Shenzhen Stock Exchanges, emphasizing long-term investment in technology, manufacturing, and consumer electronics. Goldman Sachs predicts a 13% CSI 300 Index rise in 2025, while relaxed policies and strong inflows are expected to boost valuations and investor confidence.""",
    """Foreign institutions expressed confidence in China's economic growth and A-share market value during meetings with Shanghai and Shenzhen Stock Exchanges, emphasizing long-term investment in technology, manufacturing, and consumer electronics. Goldman Sachs predicts a 13% CSI 300 Index rise in 2025, while relaxed policies and strong inflows are expected to boost valuations and investor confidence."""
]
candidates = [
    """Foreign institutions are optimistic about China's economic growth and A-share market investment value. Major Chinese stock exchanges continue to support foreign expansion. The Shanghai Stock Exchange recently met with eight foreign institutions to discuss optimizing stock connect programs and the qualified foreign institutional investors mechanism, aiming to facilitate easier access for foreign investors and promote the high-quality development of China's capital market. International investment banks are encouraged to bridge Chinese and global markets, contributing to a safer, more regulated, transparent, open, vibrant, and resilient Chinese capital market. The Shenzhen Stock Exchange also met with foreign securities brokerages, fund companies, and asset managers, focusing on the long-term investment value of Chinese high-end manufacturing, information technology, and consumer electronics industries. Incremental supportive policies since late September have stabilized market expectations and boosted international investor confidence. Both exchanges are committed to improving market transparency and predictability to facilitate foreign business expansion and investment. Inbound securities investment in China reported a net inflow of $93.1 billion in the first three quarters of 2024, marking four consecutive quarters of positive inflows. Experts predict the A-share market will attract around 2 trillion yuan ($273 billion) in capital inflow in 2025, driven by relaxed global monetary policies.""",
    """Foreign institutions are optimistic about China's economic growth and A-share market. The Shanghai and Shenzhen Stock Exchanges support foreign institutions expanding their presence and have met with them to discuss improving market access and development. Goldman Sachs and JP Morgan favor Chinese stocks, especially in technology and AI. Inbound securities investment reached $93.1 billion in 2024, signaling confidence in China's market.""",
    """Foreign institutions have a positive outlook on China's economic growth and the investment value of its A-share market. Major stock exchanges, including the Shanghai and Shenzhen Stock Exchanges, have expressed support for foreign institutions expanding their presence in the market. Meetings with foreign institutions have solicited opinions on optimizing stock connect programs and providing easier access for foreign investors. International investment banks like Goldman Sachs and JP Morgan Asset Management hold strategic preferences for Chinese stocks, with expectations of market growth and relaxed liquidity. Foreign institutions have noticed the long-term investment value of Chinese industries and vowed to deepen their roots in the market. Inbound securities investment in China reported positive inflows in the first three quarters of 2024, and experts predict the A-share market will attract significant capital inflow in 2025."""
]

# 设置模型为评估模式
model.eval()

# 不计算梯度
with torch.no_grad():
    # 对输入进行分词和编码
    inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    # 获取模型的输出
    res = model(**inputs).logits.flatten().tolist()

# 打印评估结果
print(res)