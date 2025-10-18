# Reproduction of Automatic Summarization of Events From Social Media 2013

## ⚛️项目结构说明

```
complete_2013Auto/
├── main.py                    # 主程序
├── data_loader.py            # 数据加载器
├── preprocessor.py           # 文本预处理器
├── dtm_model.py              # 时间衰减主题模型
├── gdtm_model.py             # 高斯时间衰减主题模型
├── summarizer.py             # 摘要生成器
├── evaluator.py              # 评估器
└── utils.py                  # 工具函数
```

## 🍼项目依赖

```
numpy
pandas
nltk
```

## 📝复现流程

1. 数据加载阶段
> 功能：从*Twitter*数据集加载原始推文数据<br><br>
> 实现：`TwitterDataLoader`类<br><br>
> 输出：包含推文文本、时间戳等信息的`DataFrame`

---

2. 数据预处理阶段
> 功能：清洗和提取推文中的关键信息<br><br>
> 实现：`TwitterPreprocessor` 类<br><br>
> 处理步骤：
> - 文本清洗（移除URL、@提及、特殊字符）
> - 名词短语提取（实现论文中的NP+LDA方法）
> - 时间戳解析和标准化
> - 停用词过滤

---

3. 主题模型训练
> 可选模型：
> - DTM (Decay Topic Model)：基于指数衰减的时间相关性模型
> - GDTM (Gaussian Decay Topic Model)：基于高斯分布的时间相关性模型（效果更优）
>核心创新：利用推文间的时间相关性来改进主题发现

---

4. 摘要生成
> 方法：从每个主题中选择困惑度最低的代表性推文<br><br>
> 输出：包含多个主题视角的事件摘要

---

## 💡使用说明

1. 准备Twitter数据集

2. 运行 main.py 选择模型类型

3. 查看生成的主题和摘要结果

4. 可选：使用评估模块验证摘要质量
