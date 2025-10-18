import numpy as np
import json
from datetime import datetime

def normalize_timestamps(timestamps):
    """归一化时间戳"""
    if not timestamps:
        return []
    
    timestamps = np.array(timestamps)
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    if max_time > min_time:
        return (timestamps - min_time) / (max_time - min_time)
    else:
        return np.zeros_like(timestamps)

def save_json(data, filename):
    """保存JSON数据"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filename):
    """加载JSON数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_timestamp(timestamp):
    """格式化时间戳"""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)

def calculate_similarity(text1, text2):
    """计算文本相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0