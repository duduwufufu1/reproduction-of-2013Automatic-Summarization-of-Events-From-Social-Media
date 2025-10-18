import json
import time
from datetime import datetime
from data_loader import TwitterDataLoader
from preprocessor import TwitterPreprocessor
from dtm_model import DecayTopicModel
from gdtm_model import GaussianDecayTopicModel
from summarizer import SummaryGenerator
from evaluator import SummaryEvaluator

def main():
    print("=" * 60)
    print("📊 完整复现: Automatic Summarization of Events From Social Media")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 数据加载
    print("\n(1). 📥 加载数据...")
    data_loader = TwitterDataLoader()
    tweets_df = data_loader.load_data()
    
    if tweets_df is None or len(tweets_df) == 0:
        print("❌ 无法加载数据，退出程序")
        return
    
    print(f"   成功加载 {len(tweets_df)} 条推文")
    
    # 2. 数据预处理
    print("\n(2). 🔧 数据预处理...")
    preprocessor = TwitterPreprocessor()
    processed_data = preprocessor.preprocess_tweets(tweets_df)
    
    if len(processed_data) == 0:
        print("❌ 预处理失败，退出程序")
        return
    
    doc_count = len(processed_data['documents'])
    print(f"   预处理完成，共 {doc_count} 条有效推文")
    
    # 提取训练数据
    documents = processed_data['documents']
    timestamps = processed_data['timestamps']
    original_texts = processed_data['original_texts']
    
    # 3. 训练模型
    print("\n(3). 🧠 训练主题模型...")
    
    # 选择模型 (DTM 或 GDTM)
    print("   请选择主题模型 (dtm/gdtm) [默认: gdtm]: ", end="")
    model_choice = input().strip().lower() or "gdtm"
    print(f"   选择模型: {model_choice}")
    
    if model_choice == "dtm":
        print("   使用 Decay Topic Model (DTM)...")
        model = DecayTopicModel(num_topics=5, alpha=0.1, beta=0.01)
    else:
        print("   使用 Gaussian Decay Topic Model (GDTM)...")
        model = GaussianDecayTopicModel(num_topics=8, alpha=0.1, beta=0.01)
    
    model.set_timestamps(timestamps)
    model.initialize(documents)
    
    # 训练模型
    training_start = time.time()
    model.gibbs_sample(documents, iterations=1000)# 论文中使用1000次迭代
    training_time = time.time() - training_start
    print(f"   模型训练完成，耗时: {training_time:.2f}秒")
    
    # 4. 显示主题
    print("\n(4). 🔍 发现的主题:")
    topic_words = model.get_topic_words(top_n=8)
    for topic_id, words in topic_words.items():
        word_list = [word for word, prob in words]
        print(f"   主题 {topic_id}: {word_list}")
    
    # 5. 生成摘要
    print("\n(5). 📝 生成摘要...")
    summarizer = SummaryGenerator(model)
    summary = summarizer.generate_summary(documents, original_texts, summary_length=8)
    
    print("\n" + "=" * 60)
    print("🎯 生成的事件摘要")
    print("=" * 60)
    
    for i, item in enumerate(summary):
        print(f"{i+1}. [主题 {item['topic']}] {item['text']}")
        print(f"   困惑度: {item['perplexity']:.4f}")
        print()
    
    # 6. 评估摘要 (可选)
    print("\n" + "*" * 60)
    print("因为没有合适的参考摘要，跳过评估步骤。分数没有实际意义。")
    print("\n(6). 📊 摘要评估...")
    evaluator = SummaryEvaluator()
    
    # 如果有参考摘要，可以进行评估
    reference_summary = {
        # 根据你的主题关键词创建相应的参考摘要
        "phone_hacking": [
            "Gordon Brown involved in phone hacking scandal investigation",
            "Gerry McCann phone records accessed by journalists illegally",
            "Phone hacking controversy involves politicians and celebrities"
        ],
        "tennis": [
            "Novak Djokovic wins Australian Open tennis tournament",
            "Djokovic known for entertaining personality and dancing",
            "Tennis star Djokovic popular with fans worldwide"
        ],
        "recipes": [
            "BBQ chicken sliders recipe with creamy coleslaw",
            "Easy chicken recipes for quick meals and parties",
            "Food bloggers share popular slider recipes online"
        ],
        "protests": [
            "Students and activists demonstrate at Yemen university",
            "Political protests occur at educational institutions",
            "Youth activists organize demonstrations for various causes"
        ],
        "crime": [
            "Vincent Tabak charged with murder of Joanna Yeates",
            "Police investigate murder case with suspect in custody",
            "Legal proceedings begin for high-profile murder case"
        ]
        }
    
    if len(reference_summary) > 0:
        candidate_summary = [item['text'] for item in summary]
        rouge_scores = evaluator.calculate_rouge(candidate_summary, reference_summary)
        print(f"   ROUGE-1: {rouge_scores['rouge-1']:.4f}")
        print(f"   ROUGE-2: {rouge_scores['rouge-2']:.4f}")
    print("*" * 60)
    total_time = time.time() - start_time
    print(f"\n✅ 完成! 总执行时间: {total_time:.2f}秒")
    
    # 7. 保存结果
    print("   保存结果...")
    save_results(summary, topic_words, model_choice)

def save_results(summary, topic_words, model_name):
    """保存结果到文件"""
    results = {
        "model": model_name,
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "topics": topic_words
    }
    
    # 转换为可序列化的格式
    serializable_results = {
        "model": results["model"],
        "generated_at": results["generated_at"],
        "summary": [
            {
                "topic": item["topic"],
                "text": item["text"],
                "perplexity": float(item["perplexity"])
            }
            for item in results["summary"]
        ],
        "topics": {
            str(topic_id): [
                {"word": word, "probability": float(prob)}
                for word, prob in words
            ]
            for topic_id, words in results["topics"].items()
        }
    }
    
    # 保存为JSON文件,文件名添加本地时间
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{model_name}_{timestamp_str}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()