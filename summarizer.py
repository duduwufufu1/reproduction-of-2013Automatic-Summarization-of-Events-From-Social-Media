import numpy as np

class SummaryGenerator:
    """实现论文中的摘要生成方法"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_summary(self, documents, original_texts, summary_length=8):
        """生成摘要 - 论文中的方法"""
        print("   为每个主题选择代表性文档...")
        
        topic_representatives = {}
        
        # 为每个主题找到困惑度最低的文档
        for topic in range(self.model.num_topics):
            best_doc_idx = None
            best_perplexity = float('inf')
            
            for doc_idx, document in enumerate(documents):
                perplexity = self.model.calculate_perplexity(document, topic)
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_doc_idx = doc_idx
            
            if best_doc_idx is not None:
                topic_representatives[topic] = {
                    'text': original_texts[best_doc_idx],
                    'perplexity': best_perplexity,
                    'doc_idx': best_doc_idx
                }
        
        # 按困惑度排序
        sorted_topics = sorted(topic_representatives.items(), 
                             key=lambda x: x[1]['perplexity'])
        
        # 生成摘要
        summary = []
        for topic, info in sorted_topics[:summary_length]:
            summary.append({
                'topic': topic,
                'text': info['text'],
                'perplexity': info['perplexity'],
                'doc_idx': info['doc_idx']
            })
        
        return summary
    
    def calculate_document_scores(self, documents, method='perplexity'):
        """计算文档分数（用于排名）"""
        scores = []
        
        for doc_idx, document in enumerate(documents):
            if method == 'perplexity':
                # 使用整体困惑度
                score = self.model.calculate_perplexity(document)
            else:
                # 默认使用文档长度作为简单指标
                score = len(document)
            
            scores.append((doc_idx, score))
        
        return scores