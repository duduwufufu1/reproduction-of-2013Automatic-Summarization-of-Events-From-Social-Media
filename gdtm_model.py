import numpy as np
import math
from dtm_model import DecayTopicModel

class GaussianDecayTopicModel(DecayTopicModel):
    """实现论文中的Gaussian Decay Topic Model (GDTM)"""
    
    def __init__(self, num_topics=5, alpha=0.1, beta=0.01):
        super().__init__(num_topics, alpha, beta)
        self.topic_time_means = None
        self.topic_time_variances = None
    
    def initialize(self, documents):
        """初始化模型参数"""
        super().initialize(documents)
        
        # 初始化时间参数
        if hasattr(self, 'timestamps') and len(self.timestamps) > 0:
            self.initialize_time_parameters()
    
    def initialize_time_parameters(self):
        """初始化时间参数"""
        # 随机初始化高斯参数
        self.topic_time_means = np.random.uniform(0.2, 0.8, self.num_topics)
        self.topic_time_variances = np.ones(self.num_topics) * 0.1
        
        # 根据方差更新衰减因子
        self.update_decay_parameters()
    
    def update_decay_parameters(self):
        """根据高斯分布的方差更新衰减因子 - 论文中的方法"""
        for k in range(self.num_topics):
            variance = self.topic_time_variances[k]
            if variance > 0:
                # 论文中的公式: δ = log(2) / (τ * √(2σ²log(2)))
                delta_t = math.sqrt(2 * variance * math.log(2))
                tau = 1.0  # 论文中的τ参数
                self.decay_factors[k] = math.log(2) / (tau * delta_t)
    
    def gibbs_sample(self, documents, iterations=100):
        """Gibbs采样 - 包含时间参数的更新"""
        print("   开始GDTM Gibbs采样...")
        
        for iteration in range(iterations):
            total_changes = 0
            
            for d, doc in enumerate(documents):
                current_time = self.timestamps[d] if hasattr(self, 'timestamps') else 0
                
                for i, (w_idx, old_topic) in enumerate(self.z[d]):
                    # 减少计数
                    self.n_dk[d, old_topic] -= 1
                    self.n_kw[old_topic, w_idx] -= 1
                    self.n_k[old_topic] -= 1
                    
                    # 计算新主题的概率（包含高斯时间概率）
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
            
            # 更新时间参数
            if iteration % 5 == 0 and hasattr(self, 'topic_time_means'):
                self.update_time_parameters()
            
            # 输出进度
            if (iteration + 1) % 10 == 0:
                ll = self.log_likelihood()
                print(f"     迭代 {iteration + 1}/{iterations}, 变化: {total_changes}, 似然: {ll:.2f}")
    
    def calculate_topic_probabilities(self, doc_idx, word_idx, current_time):
        """计算主题概率 - 包含高斯时间组件"""
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
                        p_prev = self.n_dk[prev_doc, k]
                        decay = math.exp(-self.decay_factors[k] * time_diff)
                        decay_sum += p_prev * decay
            
            # 高斯时间概率组件
            gaussian_time_prob = 1.0
            if hasattr(self, 'topic_time_means'):
                gaussian_time_prob = self.gaussian_probability(current_time, k)
            
            # 完整的主题概率
            p_topic_doc = (self.n_dk[doc_idx, k] + self.alpha + decay_sum) * gaussian_time_prob
            
            p_topics[k] = p_topic_doc * p_word_topic
        
        # 归一化
        total = np.sum(p_topics)
        if total > 0:
            p_topics = p_topics / total
        else:
            p_topics = np.ones(self.num_topics) / self.num_topics
            
        return p_topics
    
    def gaussian_probability(self, time, topic):
        """计算高斯时间概率"""
        if self.topic_time_means is None or self.topic_time_variances is None:
            return 1.0
            
        mean = self.topic_time_means[topic]
        variance = self.topic_time_variances[topic]
        
        if variance <= 0:
            return 1.0
            
        # 高斯概率密度函数
        exponent = -((time - mean) ** 2) / (2 * variance)
        return (1.0 / math.sqrt(2 * math.pi * variance)) * math.exp(exponent)
    
    def update_time_parameters(self):
        """更新时间参数"""
        if self.topic_time_means is None:
            return
            
        for k in range(self.num_topics):
            # 收集分配给该主题的所有时间戳
            topic_times = []
            for d, doc_assignments in enumerate(self.z):
                for word_idx, word_topic in doc_assignments:
                    if word_topic == k and hasattr(self, 'timestamps'):
                        topic_times.append(self.timestamps[d])
            
            if len(topic_times) > 1:
                # 更新均值和方差
                self.topic_time_means[k] = np.mean(topic_times)
                self.topic_time_variances[k] = np.var(topic_times)
        
        # 更新衰减因子
        self.update_decay_parameters()