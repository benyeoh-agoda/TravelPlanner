import os
from tqdm import tqdm
import argparse
from openai_request import build_plan_format_conversion_prompt, prompt_chatgpt, set_base_url


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    PARSING_MODEL = "gpt-5.2"
    parser.add_argument("--start", type=int, default=1, help="First query number (1-indexed, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last query number (1-indexed, inclusive). Defaults to end of dataset.")
    parser.add_argument("--base_url", type=str, default=None, help="Override OpenAI API base URL (e.g. https://my-gateway/v1). Falls back to OPENAI_API_BASE env var.")

    args = parser.parse_args()

    if args.base_url:
        set_base_url(args.base_url)

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'

    data = build_plan_format_conversion_prompt(directory=args.output_dir, set_type=args.set_type, model_name=args.model_name, strategy=args.strategy, mode=args.mode, start=args.start, end=args.end)

    # Per-query directory: one file per query keyed by global idx, so overlapping
    # and repeated runs never corrupt each other.
    parsed_dir = f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}_parsed'
    os.makedirs(parsed_dir, exist_ok=True)

    total_price = 0
    for idx, prompt in enumerate(tqdm(data)):
        global_idx = args.start + idx  # 1-based global query number
        out_path = f'{parsed_dir}/{global_idx}.txt'
        if prompt == "":
            open(out_path, 'w', encoding='utf-8').close()
            continue
        results, _, price = prompt_chatgpt("You are a helpful assistant.", index=idx, save_path=out_path,
                                           user_input=prompt, model_name=PARSING_MODEL, temperature=0)
        total_price += price

    print(f"Parsing Cost:${total_price}")
