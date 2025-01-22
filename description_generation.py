import argparse
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    args = parser.parse_args()

    prompt_dataset= utils.readJson(f'{args.dataset_dir}/players_description_{args.lang}.json')

    model_name = 'meta-llama/Llama-3.1-8B'
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", 
            load_in_8bit=True, 
            llm_int8_enable_fp32_cpu_offload=True,
            offload_folder='offload_weights')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Imposta il token di padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    with tqdm(total=len(prompt_dataset.keys()), desc="Processing players") as pbar:

        for p in prompt_dataset.keys():
            pbar.update(1)
            prompt = prompt_dataset[p]['prompt']
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.cuda()

            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask, 
                                        max_new_tokens=2000, 
                                        temperature=0.2,     # Modifica la temperatura qui
                                        top_k=20,            # Filtraggio top-k opzionale
                                        top_p=0.8,           # Nucleus sampling (top-p sampling) opzionale
                                        do_sample=True 
                                        #pad_token_id=tokenizer.eos_token_id
                                        )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("###Report generato:")[1]

            prompt_dataset[p]['description'] = generated_text
            print(prompt_dataset[p])
            break
    
    utils.writeJson(prompt_dataset, f'{args.dataset_dir}/player_description_{args.lang}_gen.json')
                
    