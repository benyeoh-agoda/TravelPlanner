"""
Convert Planner-R1 JSON output (<answer>...</answer> blocks) into the flat string
format expected by element_extraction.py and eval.py, bypassing parsing.py.

Usage:
    python convert_planner_r1.py \
        --set_type validation \
        --output_dir ../output \
        --model_name gpt-5 \
        --submission_file_dir ../submission

This replaces the parsing.py + element_extraction.py steps when using
--prompt_style planner_r1_json. The combination.py step is still required.
"""

import re
import json
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from tqdm import tqdm
from utils.dataset import load_query_data


def extract_answer_json(text: str):
    """Extract JSON array from <answer>...</answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def convert_transportation(t) -> str:
    if t == '-' or t is None:
        return '-'
    if isinstance(t, str):
        return t
    mode = t.get('mode', '')
    frm = t.get('from', '')
    to = t.get('to', '')
    if mode == 'flight':
        fn = t.get('flight_number', '')
        dep = t.get('departure_time', '')
        arr = t.get('arrival_time', '')
        return f"Flight Number: {fn}, from {frm} to {to}, Departure Time: {dep}, Arrival Time: {arr}"
    else:
        duration = t.get('duration', '')
        distance = t.get('distance', '')
        cost = t.get('cost', '')
        return f"{mode.capitalize()}, from {frm} to {to}, Duration: {duration}, Distance: {distance}, Cost: {cost}"


def resolve_city(city) -> str:
    """Return the city name to attach to meals/attractions/accommodation for this day.
    On travel days the city field is {'from': ..., 'to': ...}; the destination is used
    since meals and attractions occur in the city you arrive in."""
    if isinstance(city, dict):
        return city.get('to', '')
    if isinstance(city, str):
        return city
    return ''


def attach_city(name, city) -> str:
    """Append ', City' to a name-only string so eval.py's get_valid_name_city can parse it.
    The Planner-R1 JSON schema emits name-only values, so city is always appended."""
    if name is None or name == '-' or name == '':
        return '-'
    name = name.strip()
    if city:
        return f"{name}, {city}"
    return name


def convert_attraction(a, city: str = '') -> str:
    if a == '-' or a is None:
        return '-'
    if isinstance(a, str):
        # already a flat string; ensure trailing ';' that eval.py's split(';')[:-1] requires
        return a if a.endswith(';') else a + ';'
    if isinstance(a, list):
        return ''.join(attach_city(x, city) + ';' for x in a)
    return '-'


def convert_city(city) -> str:
    if isinstance(city, str):
        return city
    if isinstance(city, dict):
        return f"from {city.get('from', '')} to {city.get('to', '')}"
    return str(city)


def planner_r1_to_eval_format(days: list) -> list:
    """Convert Planner-R1 JSON day objects to the flat dict format eval.py expects.

    Meals, attractions, and accommodation are name-only in the Planner-R1 JSON schema
    (Appendix B.2); eval.py's get_valid_name_city() requires 'Name, City'.  We derive
    the city from the day's city field (destination city on travel days).
    """
    result = []
    for day in days:
        city = resolve_city(day.get("city", "-"))
        result.append({
            "days": day.get("days", ""),
            "current_city": convert_city(day.get("city", "-")),
            "transportation": convert_transportation(day.get("transportation", "-")),
            "breakfast": attach_city(day.get("breakfast", "-"), city),
            "attraction": convert_attraction(day.get("attraction", "-"), city),
            "lunch": attach_city(day.get("lunch", "-"), city),
            "dinner": attach_city(day.get("dinner", "-"), city),
            "accommodation": attach_city(day.get("accommodation", "-"), city),
        })
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing generated_plan_*.json files (same as tool_agents.py --output_dir)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--submission_file_dir", type=str, required=True,
                        help="Directory to write the final submission JSONL for eval.py")
    args = parser.parse_args()

    query_data_list = load_query_data(args.set_type)

    os.makedirs(args.submission_file_dir, exist_ok=True)
    submission_path = os.path.join(args.submission_file_dir,
                                   f"{args.set_type}_{args.model_name}_planner_r1_json_two-stage.jsonl")

    n_success = 0
    n_failed = 0

    with open(submission_path, 'w', encoding='utf-8') as out_f:
        for idx in tqdm(range(1, len(query_data_list) + 1)):
            plan_path = os.path.join(args.output_dir, args.set_type, f"generated_plan_{idx}.json")
            query = query_data_list[idx - 1]['query']
            reference = query_data_list[idx - 1].get('reference_information', '')

            try:
                generated = json.load(open(plan_path))
                raw_text = generated[-1].get(f'{args.model_name}_two-stage_results', '')
                days = extract_answer_json(raw_text)
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                days = None

            if days:
                converted = planner_r1_to_eval_format(days)
                n_success += 1
            else:
                converted = []
                n_failed += 1

            record = {
                "idx": idx,
                "query": query,
                "plan": converted,
                "reference_information": reference,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Written to: {submission_path}")
    print(f"Converted: {n_success}, Failed to parse: {n_failed}")
    print(f"\nNext step: cd evaluation && python eval.py --set_type {args.set_type} --evaluation_file_path {submission_path}")
