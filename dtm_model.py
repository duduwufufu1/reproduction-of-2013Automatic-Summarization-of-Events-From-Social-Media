import numpy as np
import random
import math
from collections import defaultdict

class DecayTopicModel:
    """实现论文中的Decay Topic Model (DTM)"""
    
    def __init__(self, num_topics=5, alpha=0.1, beta=0.01):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.decay_factors = [1.0] * num_topics  # 每个主题的衰减因子
    
    def set_timestamps(self, timestamps):
        """设置时间戳并归一化"""
        self.timestamps = np.array(timestamps)
        if len(self.timestamps) > 0:
            min_time = self.timestamps.min()
            max_time = self.timestamps.max()
            time_range = max_time - min_time
            if time_range > 0:
                self.timestamps = (self.timestamps - min_time) / time_range
            else:
                self.timestamps = np.zeros_like(self.timestamps)
    
    def build_vocab(self, documents):
        """构建词汇表"""
        all_words = set()
        for doc in documents:
            all_words.update(doc)
        
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def initialize(self, documents):
        """初始化模型参数"""
        self.build_vocab(documents)
        self.D = len(documents)
        
        # 初始化计数矩阵
        self.n_dk = np.zeros((self.D, self.num_topics))  # 文档-主题计数
        self.n_kw = np.zeros((self.num_topics, self.vocab_size))  # 主题-词计数
        self.n_k = np.zeros(self.num_topics)  # 主题总计数
        
        # 随机初始化主题分配
        self.z = []
        for d, doc in enumerate(documents):
            doc_z = []
            for word in doc:
                if word in self.vocab:
                    w_idx = self.vocab[word]
                    topic = random.randint(0, self.num_topics - 1)
                    doc_z.append((w_idx, topic))
                    
                    # 更新计数
                    self.n_dk[d, topic] += 1
                    self.n_kw[topic, w_idx] += 1
                    self.n_k[topic] += 1
            self.z.append(doc_z)
    
    def gibbs_sample(self, documents, iterations=100):
        """Gibbs采样 - 论文中的完整推理过程"""
        print("   开始Gibbs采样...")
        
        for iteration in range(iterations):
            total_changes = 0
            
            for d, doc in enumerate(documents):
                current_time = self.timestamps[d] if hasattr(self, 'timestamps') else 0
                
                for i, (w_idx, old_topic) in enumerate(self.z[d]):
                    # 减少计数
                    self.n_dk[d, old_topic] -= 1
                    self.n_kw[old_topic, w_idx] -= 1
                    self.n_k[old_topic] -= 1
                    
                    # 计算新主题的概率
                    p_topics = self.calculate_topic_probabilities(d, w_idx, current_time)
                    
                    # 采样新主题
                    new_topic = self.sample_topic(p_topics)
                    
                    # 更新计数
                    self.n_dk[d, new_topic] += 1
                    self.n_kw[new_topic, w_idx] += 1
                    self.n_k[new_topic] += 1
                    self.z[d][i] = (w_idx, new_topic)
                    
                    if new_topic != old_topic:
                        total_changes += 1
            
            # 输出进度
            if (iteration + 1) % 10 == 0:
                ll = self.log_likelihood()
                print(f"     迭代 {iteration + 1}/{iterations}, 变化: {total_changes}, 似然: {ll:.2f}")
    
    def calculate_topic_probabilities(self, doc_idx, word_idx, current_time):
        """计算主题概率 - 实现论文中的公式"""
        p_topics = np.zeros(self.num_topics)
        
        for k in range(self.num_topics):
            # P(word|topic) 部分
            p_word_topic = (self.n_kw[k, word_idx] + self.beta) / \
                          (self.n_k[k] + self.vocab_size * self.beta)
            
            # P(topic|document) with temporal decay 部分
            decay_sum = 0.0
            if hasattr(self, 'timestamps') and doc_idx > 0:
                for prev_doc in range(doc_idx):
                    time_diff = current_time - self.timestamps[prev_doc]
                    if time_diff > 0:
                        # 论文中的指数衰减: p_{i,z} * exp(-δ_z * (t_n - t_i))
                        p_prev = self.n_dk[prev_doc, k]
                        decay = math.exp(-self.decay_factors[k] * time_diff)
                        decay_sum += p_prev * decay
            
            # 完整的主题概率
            p_topic_doc = self.n_dk[doc_idx, k] + self.alpha + decay_sum
            
            p_topics[k] = p_topic_doc * p_word_topic
        
        # 归一化
        total = np.sum(p_topics)
        if total > 0:
            p_topics = p_topics / total
        else:
            # 如果所有概率都为0，使用均匀分布
            p_topics = np.ones(self.num_topics) / self.num_topics
            
        return p_topics
    
    def sample_topic(self, probs):
        """根据概率分布采样主题"""
        return np.random.choice(self.num_topics, p=probs)
    
    def log_likelihood(self):
        """计算对数似然"""
        ll = 0
        for k in range(self.num_topics):
            for w in range(self.vocab_size):
                numerator = self.n_kw[k, w] + self.beta
                denominator = self.n_k[k] + self.vocab_size * self.beta
                if numerator > 0 and denominator > 0:
                    ll += math.log(numerator / denominator)
        return ll
    
    def get_topic_words(self, top_n=10):
        """获取每个主题的关键词"""
        topic_words = {}
        for k in range(self.num_topics):
            word_probs = []
            for w_idx in range(self.vocab_size):
                prob = (self.n_kw[k, w_idx] + self.beta) / \
                      (self.n_k[k] + self.vocab_size * self.beta)
                word_probs.append((self.inv_vocab[w_idx], prob))
            
            # 按概率排序
            word_probs.sort(key=lambda x: x[1], reverse=True)
            topic_words[k] = word_probs[:top_n]
        
        return topic_words
    
    def calculate_perplexity(self, document, topic=None):
        """计算困惑度 - 论文中的评估方法"""
        if not document:
            return float('inf')
            
        log_prob = 0
        word_count = 0
        
        for word in document:
            if word in self.vocab:
                word_idx = self.vocab[word]
                
                if topic is None:
                    # 整体困惑度
                    word_prob = 0
                    for t in range(self.num_topics):
                        p_word = (self.n_kw[t, word_idx] + self.beta) / \
                                (self.n_k[t] + self.vocab_size * self.beta)
                        word_prob += p_word
                    word_prob /= self.num_topics
                else:
                    # 特定主题的困惑度
                    p_word = (self.n_kw[topic, word_idx] + self.beta) / \
                            (self.n_k[topic] + self.vocab_size * self.beta)
                    word_prob = p_word
                
                if word_prob > 0:
                    log_prob += math.log(word_prob)
                word_count += 1
        
        if word_count == 0:
            return float('inf')
            
        return math.exp(-log_prob / word_count)