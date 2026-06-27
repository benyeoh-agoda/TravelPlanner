import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from tqdm import tqdm
import json
from utils.dataset import load_query_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'

    # Per-query directory written by parsing.py
    parsed_dir = f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}_parsed'

    query_data_list = load_query_data(args.set_type)

    _end = args.end if args.end is not None else len(query_data_list)
    idx_number_list = [i for i in range(args.start, _end + 1)]
    for idx in tqdm(idx_number_list[:]):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))
        if generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_results'] not in ["", "Max Token Length Exceeded."]:
            parsed_path = f'{parsed_dir}/{idx}.txt'
            try:
                line = open(parsed_path, 'r', encoding='utf-8').read().strip()
            except FileNotFoundError:
                print(f"{idx}: parsed file missing at {parsed_path}. Run parsing.py first.")
                break
            if not line:
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
            else:
                try:
                    if '```json' in line:
                        result = line.split('```json')[1].split('```')[0]
                    else:
                        result = line.split('\t', 1)[1] if '\t' in line else line
                except Exception:
                    print(f"{idx}:\n{line}\nThis plan cannot be parsed. The plan has to follow the format ```json [The generated json format plan]```(The common gpt-4-preview-1106 json format). Please modify it manually when this occurs.")
                    break
                try:
                    generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = eval(result)
                except Exception:
                    print(f"{idx}:\n{result}\n This is an illegal json format. Please modify it manually when this occurs.")
                    break
        else:
            generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None

        with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json', 'w') as f:
            json.dump(generated_plan, f)
