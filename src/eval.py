import json
import numpy as np
from rouge_score import rouge_scorer
from collections import defaultdict
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
import torch
from tqdm import tqdm


### ======= 各評価指標の関数定義 ======= ###

def calculate_bleu(gt, gen):
    """BLEUスコアを計算"""
    return sentence_bleu([gt.split()], gen.split(), smoothing_function=SmoothingFunction().method1)


def calculate_rouge(gt, gen):
    """ROUGEスコアを計算"""
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = rouge.score(gt, gen)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }

def calculate_meteor(gt, gen):
    """METEORスコアを計算"""
    gt_tokens = word_tokenize(gt)
    gen_tokens = word_tokenize(gen)
    return meteor_score([gt_tokens], gen_tokens)

def calculate_cider(gt_abstracts, gen_abstracts):
    """CIDErスコアを計算"""
    cider = Cider()
    gts = {i: [gt] for i, gt in enumerate(gt_abstracts)}
    gens = {i: [gen] for i, gen in enumerate(gen_abstracts)}
    score, _ = cider.compute_score(gts, gens)
    return score

def calculate_bert_score(gt_abstracts, gen_abstracts):
    """BERTScore を計算"""
    P, R, F1 = bert_score(
        gen_abstracts,
        gt_abstracts,
        model_type="microsoft/deberta-xlarge",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1
    )
    return np.mean(F1.cpu().numpy())

def calculate_distinct_n(generated_texts, n=2):
    """Distinct-n（多様性評価）を計算"""
    n_gram_set = set()
    total_n_grams = 0
    for text in generated_texts:
        tokens = text.split()
        n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        n_gram_set.update(n_grams)
        total_n_grams += len(n_grams)
    
    return len(n_gram_set) / total_n_grams if total_n_grams > 0 else 0

def calculate_self_bleu(generated_texts):
    """
    Self-BLEU を計算
    """
    self_bleu_scores = []
    for i, ref in enumerate(generated_texts):
        other_texts = generated_texts[:i] + generated_texts[i+1:]  # 自分以外を参照集合に
        score = sentence_bleu([t.split() for t in other_texts], ref.split(), smoothing_function=SmoothingFunction().method1)
        self_bleu_scores.append(score)
    
    return np.mean(self_bleu_scores) if self_bleu_scores else 0

### ======= 評価 ======= ###

def evaluate_metrics(gt_abstracts, gen_abstracts):
    """全ての評価指標を計算して統合"""
    results = defaultdict(list)

    for gt, gen in tqdm(zip(gt_abstracts, gen_abstracts)):
        results["BLEU-4"].append(calculate_bleu(gt, gen))
        results["METEOR"].append(calculate_meteor(gt, gen))

        rouge_scores = calculate_rouge(gt, gen)
        results["ROUGE-1"].append(rouge_scores["ROUGE-1"])
        results["ROUGE-2"].append(rouge_scores["ROUGE-2"])
        results["ROUGE-L"].append(rouge_scores["ROUGE-L"])

    results["CIDEr"] = calculate_cider(gt_abstracts, gen_abstracts)
    results["BERTScore"] = calculate_bert_score(gt_abstracts, gen_abstracts)
    results["Distinct-2"] = calculate_distinct_n(gen_abstracts, n=2)
    results["Distinct-3"] = calculate_distinct_n(gen_abstracts, n=3)

    avg_results = {k: np.mean(v) if isinstance(v, list) else v for k, v in results.items()}
    return avg_results


### ======= 実行スクリプト ======= ###

if __name__ == "__main__":
    # データ読み込み
    with open("datasets/IIP2_test_with_generated.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON デコードエラー: {e}")
            print(f"エラー発生位置: {e.lineno} 行目, {e.colno} 列目")

    # Ground Truth & Generated Abstractのリスト作成
    gt_abstracts = [entry["abstract"] for entry in data if "abstract" in entry and "Generated_abstract" in entry]
    gen_abstracts = [entry["Generated_abstract"] for entry in data if "Generated_abstract" in entry]

    # 評価
    scores = evaluate_metrics(gt_abstracts[:10], gen_abstracts[:10])
    scores = {k: float(v) if isinstance(v, np.float32) else v for k, v in scores.items()}

    # 結果表示
    print("=== Evaluation Results ===")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    # 結果をJSONとして保存
    with open("datasets/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)
