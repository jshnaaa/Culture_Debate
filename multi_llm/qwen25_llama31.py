import torch
import datetime
import jsonlines
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub.hf_api import HfFolder
from prompt import prompts
from utils import country_capitalized_mapping, parse_final_answer, parse_response


own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

# 智能路径选择：本地路径优先，然后回退到HuggingFace Hub
def select_model_path(local_path, hub_path, model_name):
    if os.path.exists(local_path):
        print(f"使用本地路径加载{model_name}: {local_path}")
        return local_path
    else:
        print(f"本地路径不存在，使用HuggingFace Hub加载{model_name}: {hub_path}")
        return hub_path

# Qwen2.5路径选择
qwen_local_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
qwen_hub_path = "Qwen/Qwen2.5-7B-Instruct"
model1_id = select_model_path(qwen_local_path, qwen_hub_path, "Qwen2.5")

# LLaMA3.1路径选择
llama_local_path = "/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
llama_hub_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model2_id = select_model_path(llama_local_path, llama_hub_path, "LLaMA3.1")


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    # =========================================== Model Loading ===========================================
    print("正在加载Qwen2.5模型...")
    tokenizer1 = AutoTokenizer.from_pretrained(model1_id, cache_dir=own_cache_dir)
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_id,
        torch_dtype=torch.bfloat16,
        cache_dir=own_cache_dir,
        device_map="auto",
    )

    print("正在加载LLaMA3.1模型...")
    tokenizer2 = AutoTokenizer.from_pretrained(model2_id, cache_dir=own_cache_dir)
    model2 = AutoModelForCausalLM.from_pretrained(
        model2_id,
        torch_dtype=torch.bfloat16,
        cache_dir=own_cache_dir,
        device_map="auto",
    )

    print("模型加载完成，开始处理数据...")

    # =========================================== Load Dataset ===========================================
    generations1, generations2 = [], []

    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rule = line['Rule-of-Thumb']
                gold_label = line["Gold Label"]

                print(f"\n处理国家: {country}")
                print(f"故事: {story[:100]}...")

                # [1] Model1 (Qwen2.5) - Make initial decision
                print("[1] Qwen2.5 - Make initial decision")
                prompt1_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt1_1)

                # 尝试使用chat template，如果不支持则使用直接编码
                try:
                    messages1 = [{"role": "user", "content": prompt1_1}]
                    input_ids1 = tokenizer1.apply_chat_template(messages1, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                    outputs1 = model1.generate(input_ids1, max_new_tokens=1024, do_sample=False, temperature=0.0)
                    response1 = outputs1[0][input_ids1.shape[-1]:]
                    generation1_1 = tokenizer1.decode(response1, skip_special_tokens=True)
                except:
                    # 回退到直接编码方式
                    input_ids1 = tokenizer1(prompt1_1, return_tensors="pt").to(model1.device)
                    outputs1 = model1.generate(**input_ids1, max_new_tokens=1024, temperature=0.0)
                    generation1_1 = tokenizer1.decode(outputs1[0])

                generation1_1 = parse_response(generation1_1, "Answer:")
                print(f">>> {generation1_1}")
                print("-----------------------------------------------------")

                # [2] Model2 (LLaMA3.1) - Make initial decision
                print("[2] LLaMA3.1 - Make initial decision")
                prompt2_1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule)
                print(prompt2_1)
                messages2 = [{"role": "user", "content": prompt2_1}]
                input_ids2 = tokenizer2.apply_chat_template(messages2, add_generation_prompt=True, return_tensors="pt").to(model2.device)

                # LLaMA3.1的特殊terminators
                terminators = [tokenizer2.eos_token_id]
                if tokenizer2.convert_tokens_to_ids("<|eot_id|>") != tokenizer2.unk_token_id:
                    terminators.append(tokenizer2.convert_tokens_to_ids("<|eot_id|>"))

                outputs2 = model2.generate(input_ids2, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0)
                response2 = outputs2[0][input_ids2.shape[-1]:]
                generation2_1 = tokenizer2.decode(response2, skip_special_tokens=True)
                generation2_1 = parse_response(generation2_1, "Answer:")
                print(f">>> {generation2_1}")
                print("-----------------------------------------------------")

                # [3] Model1 (Qwen2.5) - Give feedback to LLaMA3.1
                print("[3] Qwen2.5 - Give feedback to LLaMA3.1")
                prompt1_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation2_1).replace("{{your_response}}", generation1_1)
                print(prompt1_2)

                try:
                    messages1_2 = [{"role": "user", "content": prompt1_2}]
                    input_ids1_2 = tokenizer1.apply_chat_template(messages1_2, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                    outputs1_2 = model1.generate(input_ids1_2, max_new_tokens=1024, do_sample=False, temperature=0.0)
                    response1_2 = outputs1_2[0][input_ids1_2.shape[-1]:]
                    generation1_2 = tokenizer1.decode(response1_2, skip_special_tokens=True)
                except:
                    input_ids1_2 = tokenizer1(prompt1_2, return_tensors="pt").to(model1.device)
                    outputs1_2 = model1.generate(**input_ids1_2, max_new_tokens=1024, temperature=0.0)
                    generation1_2 = tokenizer1.decode(outputs1_2[0])

                generation1_2 = parse_response(generation1_2, "Response:")
                print(f">>> {generation1_2}")
                print("-----------------------------------------------------")

                # [4] Model2 (LLaMA3.1) - Give feedback to Qwen2.5
                print("[4] LLaMA3.1 - Give feedback to Qwen2.5")
                prompt2_2 = prompts["prompt_2"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_response}}", generation1_1).replace("{{your_response}}", generation2_1)
                print(prompt2_2)
                messages2_2 = [{"role": "user", "content": prompt2_2}]
                input_ids2_2 = tokenizer2.apply_chat_template(messages2_2, add_generation_prompt=True, return_tensors="pt").to(model2.device)
                outputs2_2 = model2.generate(input_ids2_2, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0)
                response2_2 = outputs2_2[0][input_ids2_2.shape[-1]:]
                generation2_2 = tokenizer2.decode(response2_2, skip_special_tokens=True)
                generation2_2 = parse_response(generation2_2, "Response:")
                print(f">>> {generation2_2}")
                print("-----------------------------------------------------")

                # [5] Model1 (Qwen2.5) - Make final decision
                print("[5] Qwen2.5 - Make final decision")
                prompt1_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_feedback}}", generation2_2).replace("{{your_feedback}}", generation1_2).replace("{{your_response}}", generation1_1).replace("{{other_response}}", generation2_1)
                print(prompt1_3)

                try:
                    messages1_3 = [{"role": "user", "content": prompt1_3}]
                    input_ids1_3 = tokenizer1.apply_chat_template(messages1_3, add_generation_prompt=True, return_tensors="pt").to(model1.device)
                    outputs1_3 = model1.generate(input_ids1_3, max_new_tokens=1024, do_sample=False, temperature=0.0)
                    response1_3 = outputs1_3[0][input_ids1_3.shape[-1]:]
                    generation1_3 = tokenizer1.decode(response1_3, skip_special_tokens=True)
                except:
                    input_ids1_3 = tokenizer1(prompt1_3, return_tensors="pt").to(model1.device)
                    outputs1_3 = model1.generate(**input_ids1_3, max_new_tokens=1024, temperature=0.0)
                    generation1_3 = tokenizer1.decode(outputs1_3[0])

                generation1_3 = parse_final_answer(generation1_3)
                print(f">>> {generation1_3}")
                print("-----------------------------------------------------")

                # [6] Model2 (LLaMA3.1) - Make final decision
                print("[6] LLaMA3.1 - Make final decision")
                prompt2_3 = prompts["prompt_3"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rule}}", rule).replace("{{other_feedback}}", generation1_2).replace("{{your_feedback}}", generation2_2).replace("{{your_response}}", generation2_1).replace("{{other_response}}", generation1_1)
                print(prompt2_3)
                messages2_3 = [{"role": "user", "content": prompt2_3}]
                input_ids2_3 = tokenizer2.apply_chat_template(messages2_3, add_generation_prompt=True, return_tensors="pt").to(model2.device)
                outputs2_3 = model2.generate(input_ids2_3, max_new_tokens=1024, eos_token_id=terminators, do_sample=False, temperature=0.0)
                response2_3 = outputs2_3[0][input_ids2_3.shape[-1]:]
                generation2_3 = tokenizer2.decode(response2_3, skip_special_tokens=True)
                generation2_3 = parse_final_answer(generation2_3)
                print(f">>> {generation2_3}")
                print("-----------------------------------------------------")

                print(f"Gold > {gold_label}")
                print(f"Qwen2.5 > {generation1_3}")
                print(f"LLaMA3.1 > {generation2_3}")
                print("\n======================================================\n")
                generations1.append(generation1_3)
                generations2.append(generation2_3)

                # 保存结果
                line[f"qwen25_1"] = generation1_1
                line[f"llama31_1"] = generation2_1
                line[f"qwen25_2"] = generation1_2
                line[f"llama31_2"] = generation2_2
                line[f"qwen25_final"] = generation1_3
                line[f"llama31_final"] = generation2_3
                outfile.write(line)

    end_time = datetime.datetime.now()
    print(f"处理完成，总耗时: {end_time - start_time}")


if __name__ == "__main__":
    main()