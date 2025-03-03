import argparse
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate player and team descriptions")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the dataset file.")
    parser.add_argument("--lang", type=str, required=True, choices=["it", "en"], help="Language option. Choose between 'it' (Italian) or 'en' (English).")
    parser.add_argument("--is_player", type=int, required=False, default=1, help="Choose if run generation for players or teams.")
    args = parser.parse_args()

    #vediamo con few shot prompting distillation
    if args.is_player:
        #filename = f'{args.dataset_dir}/players_description_{args.lang}_v4_post.json'
        filename = f'{args.dataset_dir}/prova_stats_v2.json'

    else: 
        filename = f'{args.dataset_dir}/team_description_{args.lang}.json'

    prompt_dataset= utils.readJson(filename)

    #use_gpu = torch.cuda.is_available()  # Verifica se c'è una GPU disponibile
    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    #model_name = 'meta-llama/Llama-3.1-8B'
    #model_name = 'meta-llama/Llama-3.2-3B'
    #model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    model_name = 'Qwen/Qwen2.5-7B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True, 
            llm_int8_enable_fp32_cpu_offload=True,  # Solo se è su CPU
            offload_folder='offload_weights'  # Solo se è su CPU
            )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Imposta il token di padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
    count = 0
    with tqdm(total=len(prompt_dataset.keys()), desc="Processing players") as pbar:

        for p in prompt_dataset.keys():
            pbar.update(1)

            #effettuo operazioni solo se non ho già generato descrizione per il giocatore nel caso di run multiple
            if 'description' not in prompt_dataset[p].keys():
                prompts = prompt_dataset[p]['prompt']
                prompt_dataset[p]['description'] = {}

                for k, prompt in prompts.items():
                    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
                    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.to(device)

                    with torch.no_grad():
                        outputs = model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask, 
                                                max_new_tokens=300, 
                                                temperature=0.7,     # Modifica la temperatura qui
                                                top_k=20,            # Filtraggio top-k opzionale
                                                top_p=0.8,           # Nucleus sampling (top-p sampling) opzionale
                                                do_sample=True, 
                                                #pad_token_id=tokenizer.eos_token_id
                                                repetition_penalty=1.05
                                                ,pad_token_id=tokenizer.eos_token_id
                                                #,eos_token_id=tokenizer.convert_tokens_to_ids("[FINE REPORT]")

                                                )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("###Generated Report:")[1]
                    prompt_dataset[p]['description'][k] = generated_text

                print(p, prompt_dataset[p]['description'].keys())
                summary_prompt = f"""You are a professional soccer analyst specializing in player scouting and team strategy.
                ##Task
                Generate a concise and coherent summary (150 tokens) that encapsulates the overall playing style and strengths/weaknesses of a player based on performance areas:
                {list(prompt_dataset[p]['description'].keys())}
                ##Instructions
                Synthesize the provided descriptions into a unified scouting report.
                Avoid listing individual sections separately—integrate the information naturally.
                Maintain a professional tone, focusing on interpretation rather than raw data.
                Do not include references to external comparisons, statistics, or missing data.
                Ensure the summary is cohesive, presenting the player as a complete profile rather than a segmented analysis.
                Use [END_REPORT] when you end the description
                The report must focus on charateristics related to the position (e.g defender should focus on defensive skills)

                ##Input
                Position: {prompt_dataset[p]['position']}
                """
                for k, desc in prompt_dataset[p]['description'].items():
                    summary_prompt += f'{k}: {desc}\n'

                summary_prompt+= "###Generated Report:"

                input_ids = tokenizer(summary_prompt, return_tensors="pt", truncation=True).input_ids.to(device)
                attention_mask = tokenizer(summary_prompt, return_tensors="pt", padding=True, truncation=True).attention_mask.to(device)

                with torch.no_grad():
                        summary_outputs = model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask, 
                                                max_new_tokens=300, 
                                                temperature=1,     # Modifica la temperatura qui
                                                top_k=20,            # Filtraggio top-k opzionale
                                                top_p=0.8,           # Nucleus sampling (top-p sampling) opzionale
                                                do_sample=True, 
                                                #pad_token_id=tokenizer.eos_token_id
                                                repetition_penalty=1.2
                                                ,pad_token_id=tokenizer.eos_token_id
                                                #,eos_token_id=tokenizer.convert_tokens_to_ids("[FINE REPORT]")
                                                )
                print(len(tokenizer.decode(summary_outputs[0], skip_special_tokens=True).split("###Generated Report:")))
                summary_text = tokenizer.decode(summary_outputs[0], skip_special_tokens=True).split("###Generated Report:")[1]
                prompt_dataset[p]['description']['summary'] = summary_text
                print(p, prompt_dataset[p]['description'].keys())

                count+=1


            #ogni 10 descrizioni generate aggiorno il file target
            if count % 10 == 0:
                utils.writeJson(prompt_dataset, filename)
            

    
    utils.writeJson(prompt_dataset, filename)
                
    