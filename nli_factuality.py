import torch
import numpy as np
import nltk
import sys # 新增：用于退出程序
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset 

# --- NLTK 自动下载检查函数 ---
def check_and_download_nltk_resources():
    """
    检查并自动下载 NLTK 所需的 'punkt' 和 'punkt_tab' 资源。
    """
    required_resources = ['punkt', 'punkt_tab']
    all_ok = True
    
    print("Checking NLTK resources...")
    for resource in required_resources:
        try:
            # 尝试查找资源，如果找不到会抛出 LookupError
            nltk.data.find(f'tokenizers/{resource}')
            print(f"NLTK Resource '{resource}' is available.")
        except nltk.downloader.DownloadError:
             print(f"NLTK Resource '{resource}' not found. Attempting automatic download...")
             try:
                 nltk.download(resource)
                 print(f"NLTK Resource '{resource}' downloaded successfully.")
             except Exception as e:
                 print(f"CRITICAL ERROR: Failed to download NLTK resource '{resource}'.")
                 print(f"Error: {e}")
                 print("Please ensure you have network access or run 'python -m nltk.downloader {resource}' manually.")
                 all_ok = False
    
    if not all_ok:
        # 如果下载失败，退出程序，避免后续崩溃
        sys.exit(1)
    print("-" * 50)


class NLIFactualityMetric:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        """
        初始化 NLI 事实性评估指标。
        """
        
        # 在加载模型之前，首先检查并下载 NLTK 资源
        check_and_download_nltk_resources() # <--- 在这里调用自动检查和下载
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model: {model_name} on {self.device}...")
        
        # 1. 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 2. 确定蕴含 (entailment) 标签的索引
        self.entailment_idx = self.model.config.label2id.get('entailment')
        if self.entailment_idx is None:
            self.entailment_idx = 2 
            print("Warning: Could not find 'entailment' in label2id, assuming index 2.")

    # ... (score_single_pair, evaluate_summary, evaluate_batch, 以及 if __name__ == "__main__": 保持不变) ...
    # 为了完整性，这里只显示修改的部分，请确保您在本地保留所有方法。
    
    def score_single_pair(self, premise, hypothesis):
        """计算单句假设对前提的蕴含分数"""
        
        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
        return probs[0][self.entailment_idx].item()

    def evaluate_summary(self, source_doc, summary):
        """
        评估摘要的事实一致性。
        """
        # 1. 分句
        summary_sents = nltk.tokenize.sent_tokenize(summary)
        if not summary_sents:
            return 0.0
        
        scores = []
        for sent in summary_sents:
            score = self.score_single_pair(source_doc, sent)
            scores.append(score)
            
        # 2. 聚合分数 (取平均值)
        return np.mean(scores)

    def evaluate_batch(self, examples):
        """
        批量评估。
        """
        results = []
        total = len(examples)
        for i, ex in enumerate(examples):
            score = self.evaluate_summary(ex['source'], ex['summary'])
            results.append(score)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{total} examples. Current mean score: {np.mean(results):.4f}")
        return results

# --- 使用示例 --- (请确保 main.py 不再重复定义，这里只是一个独立运行的示例)
if __name__ == "__main__":
    
    metric = NLIFactualityMetric(device='cpu') # 强制使用 CPU 进行简单测试
    
    # 模拟数据测试
    test_data = [
        {
            "source": "Joe Biden met with congressional leaders at the White House today to discuss a new economic relief package.",
            "summary": "Joe Biden visited the White House today." 
        },
        # ... (省略)
    ]

    print("\n=== Running Sanity Check Tests ===")
    scores = metric.evaluate_batch(test_data)
    print(f"Overall Mean Score: {np.mean(scores):.4f}")