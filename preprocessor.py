import re
import nltk
import pandas as pd
from datetime import datetime
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

class TwitterPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # 添加Twitter特定的停用词
        self.stop_words.update(['rt', 'http', 'https', 'com', 'www', 'html', 'co'])
    
    def preprocess_tweets(self, tweets_df, max_tweets=500):#修改推文数量限制，默认500
        """预处理推文数据 - 修复版本"""
        processed_data = {
            'documents': [],
            'timestamps': [],
            'original_texts': []
        }
    
        # 限制推文数量
        tweets_df = tweets_df.head(max_tweets)
    
        processed_count = 0
        skipped_count = 0
    
        for idx, row in tweets_df.iterrows():
            # 提取文本
            text = self.extract_text(row)
        
            # 提取时间戳
            timestamp = self.extract_timestamp(row)
        
            # 提取名词短语
            noun_phrases = self.extract_noun_phrases(text)
        
            if noun_phrases and len(noun_phrases) >= 2:  # 至少2个名词短语
                processed_data['documents'].append(noun_phrases)
                processed_data['timestamps'].append(timestamp)
                processed_data['original_texts'].append(text)
                processed_count += 1
            else:
                skipped_count += 1
    
        print(f"   成功预处理 {processed_count} 条推文")
        print(f"   跳过 {skipped_count} 条无效推文")
        print(f"   文档数量: {len(processed_data['documents'])}")
        print(f"   时间戳数量: {len(processed_data['timestamps'])}")
        print(f"   原始文本数量: {len(processed_data['original_texts'])}")
    
        # 验证数据一致性
        if not (len(processed_data['documents']) == len(processed_data['timestamps']) == len(processed_data['original_texts'])):
            print("   ⚠️ 警告: 数据长度不一致!")
            # 修复数据不一致问题
            min_length = min(len(processed_data['documents']), 
                            len(processed_data['timestamps']), 
                            len(processed_data['original_texts']))
        
            processed_data['documents'] = processed_data['documents'][:min_length]
            processed_data['timestamps'] = processed_data['timestamps'][:min_length]
            processed_data['original_texts'] = processed_data['original_texts'][:min_length]
            print(f"   已修复为统一长度: {min_length}")
    
        return processed_data
    
    def extract_text(self, tweet):
        """提取推文文本"""
        if isinstance(tweet, dict):
            for field in ['full_text', 'body', 'content','textCleaned']:
                if field in tweet and isinstance(tweet[field], str):
                    return tweet[field]
            # 如果找不到文本字段，尝试转换为字符串
            return str(tweet)
        elif hasattr(tweet, 'text'):
            return tweet.text
        else:
            return str(tweet)
    
    def extract_timestamp(self, tweet):
        """提取时间戳"""
        if isinstance(tweet, dict):
            for field in ['created_at', 'timestamp', 'date', 'time']:
                if field in tweet:
                    timestamp = tweet[field]
                    return self.parse_timestamp(timestamp)
        
        # 默认返回当前时间
        return datetime.now().timestamp()
    
    def parse_timestamp(self, timestamp):
        """解析时间戳"""
        try:
            if isinstance(timestamp, (int, float)):
                return float(timestamp)
            elif isinstance(timestamp, str):
                # 尝试解析常见的时间格式
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%d %H:%M:%S',
                    '%a %b %d %H:%M:%S %z %Y',  # Twitter格式
                    '%Y-%m-%d'
                ]
                
                for fmt in formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.timestamp()
                    except ValueError:
                        continue
                
                # 如果所有格式都失败，尝试其他方法
                if 'T' in timestamp and 'Z' in timestamp:
                    timestamp = timestamp.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(timestamp)
                    return dt.timestamp()
        except Exception as e:
            print(f"   时间戳解析错误: {e}")
        
        # 默认返回当前时间
        return datetime.now().timestamp()
    
    def extract_noun_phrases(self, text):
        """提取名词短语 - 实现论文中的NP+LDA方法"""
        try:
            # 清洗文本
            text = self.clean_text(text)
            
            # 分词和词性标注
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # 提取名词短语 (论文中的Base_NP和Conj_NP)
            noun_phrases = []
            current_phrase = []
            
            for word, pos in pos_tags:
                word_lower = word.lower()
                
                # 跳过停用词和短词
                if word_lower in self.stop_words or len(word) < 3:
                    if current_phrase:
                        phrase = ' '.join(current_phrase)
                        noun_phrases.append(phrase)
                        current_phrase = []
                    continue
                
                # 名词短语模式: 形容词* 名词+
                if pos.startswith(('NN', 'JJ')):  # 名词或形容词
                    current_phrase.append(word_lower)
                elif current_phrase:
                    # 结束当前短语
                    phrase = ' '.join(current_phrase)
                    noun_phrases.append(phrase)
                    current_phrase = []
            
            # 处理最后一个短语
            if current_phrase:
                phrase = ' '.join(current_phrase)
                noun_phrases.append(phrase)
            
            return noun_phrases
            
        except Exception as e:
            print(f"   名词短语提取错误: {e}")
            # 备用方法: 简单的关键词提取
            return self.fallback_keyword_extraction(text)
    
    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            text = str(text)
        
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        # 移除@提及
        text = re.sub(r'@\w+', '', text)
        # 移除#符号但保留标签文本
        text = re.sub(r'#', '', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fallback_keyword_extraction(self, text):
        """备用关键词提取方法"""
        try:
            text = self.clean_text(text)
            words = word_tokenize(text.lower())
            
            # 过滤停用词和短词
            keywords = [
                word for word in words 
                if word not in self.stop_words and len(word) >= 4
            ]
            
            return keywords[:10]  # 限制数量
            
        except:
            # 最后的手段: 简单的分词
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            return [w for w in words if w not in self.stop_words][:8]