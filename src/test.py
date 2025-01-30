import torch
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm

### ======= モデルの読み込みとデータセットの読み込み ======= ###

# モデルの読み込み
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    model_name="checkpoints/checkpoint-100",
    max_seq_length=8192,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# データセットの読み込み
test_dataset = pd.read_json("datasets/test.json", dtype={"id": str})

# プロンプトテンプレートの定義
prompt_template = """
### Instruction:
Write a abstract for the following scientific paper.

### Input:
{} 

### Output:
"""

### ======= 生成 ======= ###

torch.cuda.empty_cache()
FastLanguageModel.for_inference(model)
model.eval()

results = []

with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset))):
        # プロンプトを生成
        sample = test_dataset.iloc[i]
        full_text = " ".join(sample["Full-Text"].split()[:6000])
        prompt = tokenizer(
            [
                prompt_template.format(full_text)
            ],
            return_tensors = "pt"
        ).to("cuda")

        # 終了トークンの設定
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 推論
        outputs = model.generate(
            **prompt, 
            max_new_tokens = 256,
            use_cache = True,
            eos_token_id=terminators,
            temperature=0.6,
            top_p=0.9
        )

        # デコード
        decoded = tokenizer.batch_decode(outputs,skip_special_tokens=True)
        decoded = decoded[0].split("### Output:")[1]

        # 結果をリストに追加
        results.append(
            {
                "id": sample["id"],
                'title': sample['title'],
                'abstract': sample['abstract'],
                "Generated_abstract": decoded.strip(),
                'Full-Text': sample['Full-Text'],
                'GA_path': sample['GA_path'],
                'GA_caption': sample['GA_caption'],
                'GA_components': sample['GA_components'],
                'Fig.n_path': sample['Fig.n_path'],
                'Fig.n_caption': sample['Fig.n_caption'],
                'subjects': sample['subjects'],
                'journal_ref': sample['journal_ref'],
                'conference': sample['conference'],
            }
        )

### ======= 結果を JSON に保存 ======= ###

result_df = pd.DataFrame(results)
result_df.to_json("outputs/IIP2_test_with_generated_ft_2.json", orient="records", indent=4, force_ascii=False)

