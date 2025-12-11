import torch
import numpy as np
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from nli_factuality import NLIFactualityMetric
import time

# --- 配置 ---
MODEL_HUB_ID = "mercuryujia/bart-large-multi-news"
DATASET_NAME = "Awesome075/multi_news_parquet"
SPLIT_NAME = "test"
# 仅评估前 N 个样本以进行快速测试。如果要评估全集，请设置为 None 或更大的数字。
MAX_SAMPLES = 50 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_SIZE = 16  # 每个文档生成的摘要数量

print(f"Using device: {DEVICE}")

def generate_summaries(model, tokenizer, documents, device, max_length=150, batch_size=4, num_beams=4):
    """
    使用 BART 模型批量生成摘要。
    """
    all_generated_summaries = []
    
    # 按照 batch_size 进行分批
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # 1. 对文档进行分词
        inputs = tokenizer(
            batch, 
            max_length=1024, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # 2. 将输入移动到指定设备
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 3. 生成摘要
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,                 # <--- 修复点 1: 传入 BEAM_SIZE
                num_return_sequences=num_beams,      # <--- 修复点 2: 确保为每个输入返回 num_beams 个结果
                do_sample=False,
                length_penalty=2.0,
                early_stopping=True
            )
            
        # 4. 解码生成的 ID 为文本
        generated_summaries = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        all_generated_summaries.extend(generated_summaries)
        
        # 打印进度
        if (i + batch_size) % 10 == 0:
             print(f"[Generation] Processed {i + batch_size}/{len(documents)} examples...")
             
    return all_generated_summaries

def main():
    # 确保 BEAM_SIZE 在文件顶部被定义 (例如：BEAM_SIZE = 4)
    global BEAM_SIZE 
    start_time = time.time()
    
    # --- 阶段一：模型和数据准备 ---
    print("\n--- Phase 1: Loading Model and Data ---")
    
    # 1. 加载微调的 BART 模型和分词器
    try:
        tokenizer = BartTokenizer.from_pretrained(MODEL_HUB_ID)
        model = BartForConditionalGeneration.from_pretrained(MODEL_HUB_ID).to(DEVICE)
        model.eval()
        print(f"Successfully loaded model: {MODEL_HUB_ID}")
    except Exception as e:
        print(f"Error loading model from Hub: {e}")
        print("Please ensure the model name is correct and accessible.")
        return

    # 2. 加载测试数据集
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    
    # 限制样本数量
    if MAX_SAMPLES is not None:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
        print(f"Loaded {len(dataset)} samples for quick testing.")
    else:
        print(f"Loaded {len(dataset)} samples from the full {SPLIT_NAME} split.")

    documents = dataset['document']
    num_documents = len(documents)
    
    # --- 阶段二：摘要生成 ---
    print("\n--- Phase 2: Generating Summaries ---")
    
    # 3. 生成摘要：使用 BEAM_SIZE 作为 num_beams 的值
    # 注意：您的 generate_summaries 函数需要确保将 num_beams 设置为 BEAM_SIZE
    print(f"Generating {BEAM_SIZE} summaries per document...")
    generated_summaries = generate_summaries(model, tokenizer, documents, DEVICE, num_beams=BEAM_SIZE)
    
    # 检查摘要数量是否符合预期
    expected_count = num_documents * BEAM_SIZE
    if len(generated_summaries) != expected_count:
        print(f"ERROR: Generated summaries count ({len(generated_summaries)}) does not match expected count ({expected_count}).")
        print("Please check the 'generate_summaries' function to ensure it outputs BEAM_SIZE summaries per document.")
        return

    # --- 阶段三：NLI 事实性评估 (针对 BEAM_SIZE 个摘要取平均) ---
    print("\n--- Phase 3: NLI Factuality Evaluation ---")
    
    # 4. 初始化 NLI 评估器
    nli_scorer = NLIFactualityMetric(device=DEVICE)

    # 5. 逐个文档计算平均分
    all_document_avg_scores = [] # 存储 F_document_i
    
    print(f"\nStarting NLI evaluation, processing {num_documents} documents ({BEAM_SIZE} summaries each)...")

    for i in range(num_documents):
        doc = documents[i]
        
        # 提取当前文档的 BEAM_SIZE 个摘要
        start_idx = i * BEAM_SIZE
        current_summaries = generated_summaries[start_idx:start_idx + BEAM_SIZE]
        
        individual_scores = []
        for gen_sum in current_summaries:
            # 返回单个摘要的句子平均分
            score = nli_scorer.evaluate_summary(doc, gen_sum)
            individual_scores.append(score)
            
        # 计算这 BEAM_SIZE 个摘要的平均分，作为该文档的最终 NLI 分数
        doc_avg_nli = np.mean(individual_scores)
        all_document_avg_scores.append(doc_avg_nli)
        
        # 打印进度
        if (i + 1) % 10 == 0:
            current_overall_mean = np.mean(all_document_avg_scores)
            print(f"[Document] Processed {i+1}/{num_documents} documents. Current mean document score: {current_overall_mean:.4f}")

    # --- 阶段四：结果输出 ---
    
    mean_nli_score = np.mean(all_document_avg_scores)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("      FINAL NLI FACTUALITY SCORE REPORT      ")
    print("="*50)
    print(f"Model ID: {MODEL_HUB_ID}")
    print(f"Dataset Split: {DATASET_NAME}/{SPLIT_NAME}")
    print(f"Number of Documents Evaluated: {num_documents}")
    print(f"Total Summaries Evaluated (Beam Size): {BEAM_SIZE}")
    print(f"Total NLI Calls: {num_documents * BEAM_SIZE}")
    print("-" * 50)
    print(f"MEAN DOCUMENT NLI SCORE (Avg of {BEAM_SIZE} Summaries): {mean_nli_score:.4f}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print("="*50)

# ... (if __name__ == "__main__": main()) ...

if __name__ == "__main__":
    main()