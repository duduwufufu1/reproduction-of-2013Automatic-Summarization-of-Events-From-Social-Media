import numpy as np
from collections import Counter
import re

class SummaryEvaluator:
    """实现论文中的评估方法"""
    
    def calculate_rouge(self, candidate_summary, reference_summary, n=2):
        """计算ROUGE-N分数 - 论文中的评估指标"""
        if not candidate_summary or not reference_summary:
            return {'rouge-1': 0, 'rouge-2': 0}
        
        # 将摘要转换为n-gram集合
        candidate_ngrams = self.get_ngrams(candidate_summary, n)
        reference_ngrams = self.get_ngrams(reference_summary, n)
        
        # 计算重叠
        overlapping_ngrams = candidate_ngrams & reference_ngrams
        
        # 计算ROUGE-N
        if len(reference_ngrams) == 0:
            rouge_n = 0
        else:
            rouge_n = len(overlapping_ngrams) / len(reference_ngrams)
        
        # 计算ROUGE-1和ROUGE-2
        rouge_1 = self.calculate_rouge_n(candidate_summary, reference_summary, 1)
        rouge_2 = self.calculate_rouge_n(candidate_summary, reference_summary, 2)
        
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2
        }
    
    def get_ngrams(self, text_list, n):
        """获取n-gram集合"""
        ngrams = set()
        for text in text_list:
            words = re.findall(r'\b\w+\b', text.lower())
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.add(ngram)
        return ngrams
    
    def calculate_rouge_n(self, candidate, reference, n):
        """计算特定n的ROUGE分数"""
        candidate_ngrams = self.get_ngrams(candidate, n)
        reference_ngrams = self.get_ngrams(reference, n)
        
        if len(reference_ngrams) == 0:
            return 0
        
        overlapping = len(candidate_ngrams & reference_ngrams)
        return overlapping / len(reference_ngrams)
    
    def calculate_precision_recall(self, candidate_summary, reference_summary):
        """计算精确率和召回率"""
        candidate_words = set()
        for text in candidate_summary:
            words = re.findall(r'\b\w+\b', text.lower())
            candidate_words.update(words)
        
        reference_words = set()
        for text in reference_summary:
            words = re.findall(r'\b\w+\b', text.lower())
            reference_words.update(words)
        
        if len(candidate_words) == 0 or len(reference_words) == 0:
            return 0, 0
        
        overlapping = len(candidate_words & reference_words)
        precision = overlapping / len(candidate_words)
        recall = overlapping / len(reference_words)
        
        return precision, recall