import os
import json
import pandas as pd
from datetime import datetime

class TwitterDataLoader:
    def __init__(self):
        self.data_paths = [
            "Twitter_data",
        ]
    
    def load_data(self):
        """加载Twitter数据"""
        # 首先尝试用户的数据文件
        for data_path in self.data_paths:
            if os.path.exists(data_path):
                return self.load_from_file(data_path)
        
        # 如果找不到数据文件，使用示例数据
        print("⚠️  未找到数据文件，使用示例数据")
        return self.create_sample_data()
    
    def load_from_file(self, file_path):
        """从文件加载数据"""
        try:
            if file_path.endswith('.json'):
                return self.load_json(file_path)
            elif file_path.endswith('.jsonl'):
                return self.load_jsonl(file_path)
            else:
                # 假设是JSONL格式
                return self.load_jsonl(file_path)
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return self.create_sample_data()
    
    def load_jsonl(self, file_path):
        """加载JSONL格式数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(data)
    
    def load_json(self, file_path):
        """加载JSON格式数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # 如果是字典，尝试找到数据数组
            for key in ['tweets', 'data', 'statuses']:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        else:
            return pd.DataFrame([data])
    
    def create_sample_data(self):
        """创建Facebook IPO的示例数据"""
        sample_tweets = [
            {
                "id": 1,
                "text": "Facebook IPO announced for May 18th, great news for tech investors worldwide",
                "created_at": "2012-05-01T10:00:00Z",
                "user": {"screen_name": "technews"}
            },
            {
                "id": 2,
                "text": "FB stock price set at $38 per share, market reaction very positive today",
                "created_at": "2012-05-01T11:30:00Z",
                "user": {"screen_name": "marketwatch"}
            },
            {
                "id": 3,
                "text": "Mark Zuckerberg leads Facebook IPO with strong vision for company future",
                "created_at": "2012-05-02T09:15:00Z",
                "user": {"screen_name": "business"}
            },
            {
                "id": 4,
                "text": "Facebook IPO faces regulatory challenges and government scrutiny this week",
                "created_at": "2012-05-02T14:20:00Z",
                "user": {"screen_name": "reuters"}
            },
            {
                "id": 5,
                "text": "Investors eagerly await Facebook stock market debut next week on NASDAQ",
                "created_at": "2012-05-03T08:45:00Z",
                "user": {"screen_name": "investor"}
            },
            {
                "id": 6,
                "text": "Facebook IPO could be the biggest tech offering in market history records",
                "created_at": "2012-05-03T12:30:00Z",
                "user": {"screen_name": "finance"}
            },
            {
                "id": 7,
                "text": "Stock analysts predict strong performance for FB shares after IPO launch",
                "created_at": "2012-05-04T10:15:00Z",
                "user": {"screen_name": "analyst"}
            },
            {
                "id": 8,
                "text": "Facebook prepares for historic Nasdaq listing ceremony next Wednesday",
                "created_at": "2012-05-04T16:40:00Z",
                "user": {"screen_name": "news"}
            },
            {
                "id": 9,
                "text": "Morgan Stanley confirmed as lead underwriter for Facebook public offering",
                "created_at": "2012-05-05T13:25:00Z",
                "user": {"screen_name": "banking"}
            },
            {
                "id": 10,
                "text": "Facebook valuation estimated at $104 billion ahead of market debut",
                "created_at": "2012-05-05T15:50:00Z",
                "user": {"screen_name": "valuation"}
            },
            {
                "id": 11,
                "text": "NASDAQ technical issues cause delay in Facebook trading start time",
                "created_at": "2012-05-06T09:30:00Z",
                "user": {"screen_name": "exchange"}
            },
            {
                "id": 12,
                "text": "Facebook stock opens at $42.05, significantly above offering price target",
                "created_at": "2012-05-06T11:15:00Z",
                "user": {"screen_name": "trading"}
            },
            {
                "id": 13,
                "text": "Zuckerberg rings opening bell remotely from Facebook California headquarters",
                "created_at": "2012-05-07T08:00:00Z",
                "user": {"screen_name": "ceo"}
            },
            {
                "id": 14,
                "text": "Facebook raises $16 billion in record-breaking technology IPO event",
                "created_at": "2012-05-07T14:30:00Z",
                "user": {"screen_name": "funding"}
            },
            {
                "id": 15,
                "text": "Early investors see massive returns on Facebook stock investment today",
                "created_at": "2012-05-08T10:45:00Z",
                "user": {"screen_name": "returns"}
            }
        ]
        return pd.DataFrame(sample_tweets)