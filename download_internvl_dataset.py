from datasets import load_dataset
ds = load_dataset("OpenGVLab/InternVL-Chat-V1-2-SFT-Data", "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl")

print(ds['train'][0])