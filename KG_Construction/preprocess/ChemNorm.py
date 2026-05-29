





import pubchempy as pcp
from rdkit import Chem

def get_smiles(name):
    compounds = pcp.get_compounds(name, 'name')
    
    if compounds:
        c = compounds[0]
        raw_smiles = c.isomeric_smiles or c.canonical_smiles
        
        if raw_smiles:
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol:
                return Chem.MolToSmiles(mol)
    return None


if __name__ == "__main__":
    print(get_smiles("7990-91-2"))





import os
import json
import os
import pubchempy as pcp
from rdkit import Chem
from rdkit import RDLogger


RDLogger.logger().setLevel(RDLogger.CRITICAL)

def get_smiles(name):
    if not name or name == "无资料":
        return None
    try:

        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            raw_smiles = c.isomeric_smiles or c.canonical_smiles
            if raw_smiles:
                mol = Chem.MolFromSmiles(raw_smiles)
                if mol:
                    return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None

def process_json_folder(input_folder, output_file):
    final_data = {}
    success_count = 0
    fail_count = 0
    failed_filenames = []
    

    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    total_files = len(json_files)
    
    print(f"开始处理，共计 {total_files} 个文件...\n")

    for filename in json_files:
        file_path = os.path.join(input_folder, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            info = data.get("sections", {}).get("第一部分：化学品及企业标识", {})
            if not info:
                raise ValueError("缺少必要的 section 信息")


            item_result = {
                "中文名称": info.get("中文名称", "无资料"),
                "中文别名": info.get("中文别名", "无资料"),
                "英文名称": info.get("英文名称", "无资料"),
                "英文别名": info.get("英文别名", "无资料"),
            }
            

            smiles = None
            if item_result["英文名称"] != "无资料":
                smiles = get_smiles(item_result["英文名称"])
            
            if not smiles and item_result["英文别名"] != "无资料":
                smiles = get_smiles(item_result["英文别名"])
            

            if smiles:
                item_result["SMILES"] = smiles
                success_count += 1
            else:
                item_result["SMILES"] = "无资料"
                fail_count += 1
                failed_filenames.append(f"{filename} (未查到SMILES)")
            
            key = data.get("标题") or item_result["中文名称"]
            final_data[key] = item_result
            print(f"[{success_count + fail_count}/{total_files}] 处理完成: {key}")

        except Exception as e:
            fail_count += 1
            failed_filenames.append(f"{filename} (文件解析错误: {str(e)})")
            print(f"处理文件 {filename} 时发生严重错误")


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    

    print("\n" + "="*30)
    print("      处理任务结束报告")
    print("="*30)
    print(f"总处理文件数: {total_files}")
    print(f"处理成功个数: {success_count} (已获取SMILES)")
    print(f"处理失败个数: {fail_count} (含格式错误及未查到物质)")
    print("-" * 30)
    
    if failed_filenames:
        print("以下为失败详情列表：")
        for fail_info in failed_filenames:
            print(f" - {fail_info}")
    else:
        print("恭喜！所有文件均成功处理。")
    print("="*30)


if __name__ == "__main__":
    process_json_folder(r"数据集预处理\MSDS", "output_master.json")





import json
import os
import time
import re
import pubchempy as pcp
from rdkit import Chem
from openai import OpenAI


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "") or None,
    api_key=os.getenv("OPENAI_API_KEY", "")
)


cache = {}


def split_english_names(text):
    if not text or text == "无资料":
        return []

    parts = re.split(r"[;,]", text)
    parts = [p.strip() for p in parts if p.strip()]

    return list(set(parts))



def generate_candidates_from_cn(name_cn):
    if name_cn in cache:
        return cache[name_cn]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a chemistry expert.\n"
                        "Generate 3-5 correct English chemical names.\n"
                        "Return ONLY JSON array.\n"
                        "Example: [\"ethanol\", \"ethyl alcohol\"]"
                    )
                },
                {"role": "user", "content": name_cn}
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        if not text.startswith("["):
            candidates = [text]
        else:
            candidates = json.loads(text)

        candidates = list(set([c.strip() for c in candidates if c.strip()]))

        cache[name_cn] = candidates
        return candidates

    except Exception as e:
        print(f"[中文候选失败] {name_cn} -> {e}")
        return []



def expand_candidates_with_llm(raw_candidates):
    if not raw_candidates:
        return []

    try:
        prompt = f"""
Given these chemical names:

{raw_candidates}

Tasks:
1. Fix spelling errors
2. Normalize to standard chemical names
3. Add possible synonyms

Return ONLY a JSON array of names.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        if not text.startswith("["):
            return raw_candidates

        return list(set(json.loads(text)))

    except Exception as e:
        print(f"[候选清洗失败] -> {e}")
        return raw_candidates



def get_smiles(name):
    try:
        compounds = pcp.get_compounds(name, 'name')

        if compounds:
            c = compounds[0]
            raw = c.isomeric_smiles or c.canonical_smiles

            if raw:
                mol = Chem.MolFromSmiles(raw)
                if mol:
                    return Chem.MolToSmiles(mol)
    except:
        pass

    return None



def process_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    success, fail = [], []

    for key, value in data.items():

        if value.get("SMILES") != "无资料":
            continue

        name_cn = value.get("中文名称", key)

        print("\n====================")
        print(f"处理: {name_cn}")

        candidates = []


        eng_name = value.get("英文名称", "")
        eng_alias = value.get("英文别名", "")

        candidates += split_english_names(eng_name)
        candidates += split_english_names(eng_alias)

        if candidates:
            print(f"👉 原始英文拆分: {candidates}")


            candidates = expand_candidates_with_llm(candidates)


        cn_candidates = generate_candidates_from_cn(name_cn)
        candidates += cn_candidates


        candidates = list(set(candidates))

        print(f"👉 最终候选: {candidates}")


        found = False

        for name in candidates:
            if len(name)<4:
                continue
            smiles = get_smiles(name)
            print(f"   尝试: {name} -> {smiles}")

            if smiles:
                print(f"✅ 成功: {name}")
                print(f"   SMILES: {smiles}")

                value["英文名称"] = name
                value["SMILES"] = smiles

                success.append(name_cn)
                found = True
                break

        if not found:
            print(f"❌ 失败: {name_cn}")
            fail.append(name_cn)

        time.sleep(1)


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n====================")
    print("处理完成")
    print(f"成功: {len(success)}")
    print(f"失败: {len(fail)}")



if __name__ == "__main__":
    process_json("cleaned_chemicals.json", "output.json")





import os
import json
import re
from pathlib import Path
from collections import defaultdict




OUTPUT_JSON_PATH = r"output.json"
MSDS_DIR = r"数据集预处理\MSDS"

UPDATED_OUTPUT_PATH = r"output_with_reference.json"
REVIEW_NEEDED_PATH = r"review_needed.json"
UNMATCHED_OUTPUT_PATH = r"unmatched_output.json"





def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s


def remove_bracket_content(s: str) -> str:
    if not s:
        return ""
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\([^()]*\)", "", s)
    return normalize_text(s)


def split_multi_name(s: str):
    if not s:
        return []
    s = s.replace("；", ";").replace("，", ",").replace("、", ",").replace("/", ",")
    parts = [normalize_text(x) for x in s.split(",")]
    parts = [x for x in parts if x]
    return parts


def extract_candidate_names(msds_data: dict, file_stem: str):
    candidates = []


    if file_stem:
        candidates.append(file_stem)


    title = msds_data.get("标题", "")
    if title:
        candidates.append(title)


    sections = msds_data.get("sections", {})
    sec1 = sections.get("第一部分：化学品及企业标识", {})

    cn_name = sec1.get("中文名称", "")
    cn_alias = sec1.get("中文别名", "")

    if cn_name and cn_name != "无资料":
        candidates.append(cn_name)
    if cn_alias and cn_alias != "无资料":
        candidates.append(cn_alias)


    seen = set()
    result = []
    for x in candidates:
        nx = normalize_text(x)
        if nx and nx not in seen:
            seen.add(nx)
            result.append(x)
    return result


def build_msds_index(msds_dir):
    strict_exact_index = defaultdict(list)
    review_index = defaultdict(list)

    msds_files = list(Path(msds_dir).glob("*.json"))

    for fp in msds_files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[跳过] 读取失败: {fp} -> {e}")
            continue

        file_stem = fp.stem
        reference = data.get("reference", "")

        sections = data.get("sections", {})
        sec1 = sections.get("第一部分：化学品及企业标识", {})
        cn_name = sec1.get("中文名称", "")
        cn_alias = sec1.get("中文别名", "")
        title = data.get("标题", "")

        record = {
            "file_name": fp.name,
            "file_stem": file_stem,
            "reference": reference,
            "标题": title,
            "中文名称": cn_name,
            "中文别名": cn_alias,
        }



        exact_fields = [file_stem, title, cn_name]
        if cn_alias and cn_alias != "无资料":
            exact_fields.append(cn_alias)

        for val in exact_fields:
            nv = normalize_text(val)
            if nv:
                strict_exact_index[nv].append(record)






        raw_fields = [file_stem, title, cn_name]
        if cn_alias and cn_alias != "无资料":
            raw_fields.append(cn_alias)

        review_keys = set()

        for raw in raw_fields:
            nr = normalize_text(raw)
            if nr:
                review_keys.add(nr)


            for part in split_multi_name(raw):
                if part:
                    review_keys.add(part)


            no_bracket = remove_bracket_content(raw)
            if no_bracket:
                review_keys.add(no_bracket)


            cleaned_raw = raw.replace("（", "(").replace("）", ")")
            cleaned_raw = re.sub(r"\([^()]*\)", "", cleaned_raw)
            for part in split_multi_name(cleaned_raw):
                if part:
                    review_keys.add(part)

        for key in review_keys:
            review_index[key].append(record)

    return strict_exact_index, review_index


def deduplicate_records(records):
    seen = set()
    result = []
    for r in records:
        k = (r["file_name"], r["reference"])
        if k not in seen:
            seen.add(k)
            result.append(r)
    return result


def main():
    output_data = load_json(OUTPUT_JSON_PATH)
    strict_exact_index, review_index = build_msds_index(MSDS_DIR)

    updated_output = json.loads(json.dumps(output_data, ensure_ascii=False))
    review_needed = []
    unmatched_output = []

    direct_match_count = 0
    review_count = 0
    unmatched_count = 0

    for out_key, out_info in output_data.items():

        out_cn_name = out_key
        if isinstance(out_info, dict):
            out_cn_name = out_info.get("中文名称", out_key)

        norm_out_name = normalize_text(out_cn_name)


        exact_matches = strict_exact_index.get(norm_out_name, [])
        exact_matches = deduplicate_records(exact_matches)

        if len(exact_matches) == 1:
            updated_output[out_key]["reference"] = exact_matches[0]["reference"]
            updated_output[out_key]["matched_msds_file"] = exact_matches[0]["file_name"]
            updated_output[out_key]["match_type"] = "direct"
            direct_match_count += 1
            continue

        elif len(exact_matches) > 1:

            review_needed.append({
                "output_field": out_key,
                "output_中文名称": out_cn_name,
                "reason": "严格匹配到多个MSDS文件，需人工审查",
                "candidates": exact_matches
            })
            review_count += 1
            continue


        review_matches = review_index.get(norm_out_name, [])
        review_matches = deduplicate_records(review_matches)

        if review_matches:
            review_needed.append({
                "output_field": out_key,
                "output_中文名称": out_cn_name,
                "reason": "疑似逗号拆分或去括号后的匹配，需人工审查",
                "candidates": review_matches
            })
            review_count += 1
        else:
            unmatched_output.append({
                "output_field": out_key,
                "output_中文名称": out_cn_name
            })
            unmatched_count += 1

    save_json(updated_output, UPDATED_OUTPUT_PATH)
    save_json(review_needed, REVIEW_NEEDED_PATH)
    save_json(unmatched_output, UNMATCHED_OUTPUT_PATH)

    print("处理完成：")
    print(f"  直接匹配写回: {direct_match_count}")
    print(f"  需要人工审查: {review_count}")
    print(f"  未找到匹配:   {unmatched_count}")
    print()
    print(f"已输出：")
    print(f"  {UPDATED_OUTPUT_PATH}")
    print(f"  {REVIEW_NEEDED_PATH}")
    print(f"  {UNMATCHED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()





import os
import json
import re
from pathlib import Path
from collections import defaultdict




OUTPUT_JSON_PATH = r"output.json"
MSDS_DIR = r"数据集预处理\MSDS"

FINAL_OUTPUT_PATH = r"output_aligned.json"
UNMATCHED_OUTPUT_PATH = r"unmatched_output.json"





def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s


def remove_bracket_content(s: str) -> str:
    if not s:
        return ""
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\([^()]*\)", "", s)
    return normalize_text(s)


def split_multi_name(s: str):
    if not s:
        return []
    s = s.replace("；", ";").replace("，", ",").replace("、", ",").replace("/", ",")
    parts = [normalize_text(x) for x in s.split(",")]
    return [x for x in parts if x]


def deduplicate_records(records):
    seen = set()
    result = []
    for r in records:
        k = (r["file_name"], r.get("reference", ""))
        if k not in seen:
            seen.add(k)
            result.append(r)
    return result





def build_msds_index(msds_dir):
    strict_exact_index = defaultdict(list)
    review_index = defaultdict(list)

    msds_files = list(Path(msds_dir).glob("*.json"))

    for fp in msds_files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[跳过] 读取失败: {fp} -> {e}")
            continue

        file_stem = fp.stem
        reference = data.get("reference", "")

        sections = data.get("sections", {})
        sec1 = sections.get("第一部分：化学品及企业标识", {})

        title = data.get("标题", "")
        cn_name = sec1.get("中文名称", "")
        cn_alias = sec1.get("中文别名", "")

        record = {
            "file_name": fp.name,
            "file_stem": file_stem,
            "reference": reference,
            "标题": title,
            "中文名称": cn_name,
            "中文别名": cn_alias,
        }




        exact_fields = []

        if cn_name and cn_name != "无资料":
            exact_fields.append(cn_name)

        if cn_alias and cn_alias != "无资料":
            exact_fields.append(cn_alias)

        for val in exact_fields:
            nv = normalize_text(val)
            if nv:
                strict_exact_index[nv].append(record)





        raw_fields = []

        if file_stem:
            raw_fields.append(file_stem)
        if title:
            raw_fields.append(title)
        if cn_name and cn_name != "无资料":
            raw_fields.append(cn_name)
        if cn_alias and cn_alias != "无资料":
            raw_fields.append(cn_alias)

        review_keys = set()

        for raw in raw_fields:
            nr = normalize_text(raw)
            if nr:
                review_keys.add(nr)


            for part in split_multi_name(raw):
                if part:
                    review_keys.add(part)


            no_bracket = remove_bracket_content(raw)
            if no_bracket:
                review_keys.add(no_bracket)


            cleaned_raw = raw.replace("（", "(").replace("）", ")")
            cleaned_raw = re.sub(r"\([^()]*\)", "", cleaned_raw)
            for part in split_multi_name(cleaned_raw):
                if part:
                    review_keys.add(part)

        for key in review_keys:
            review_index[key].append(record)

    return strict_exact_index, review_index





def main():
    output_data = load_json(OUTPUT_JSON_PATH)
    strict_exact_index, review_index = build_msds_index(MSDS_DIR)

    final_output = {}
    unmatched_output = {}

    direct_match_count = 0
    review_count = 0
    unmatched_count = 0

    for out_key, out_info in output_data.items():

        current_item = json.loads(json.dumps(out_info, ensure_ascii=False))


        out_cn_name = out_key
        if isinstance(out_info, dict):
            out_cn_name = out_info.get("中文名称", out_key)

        norm_out_name = normalize_text(out_cn_name)




        exact_matches = deduplicate_records(strict_exact_index.get(norm_out_name, []))

        if len(exact_matches) == 1:
            current_item["reference"] = exact_matches[0]["reference"]
            current_item["matched_msds_file"] = exact_matches[0]["file_name"]
            current_item["match_type"] = "direct"

            final_output[out_key] = current_item
            direct_match_count += 1
            continue

        elif len(exact_matches) > 1:

            current_item["match_type"] = "review_needed"
            current_item["review_reason"] = "严格匹配到多个MSDS文件，需人工审查"
            current_item["review_candidates"] = exact_matches

            final_output[out_key] = current_item
            review_count += 1
            continue




        review_matches = deduplicate_records(review_index.get(norm_out_name, []))

        if review_matches:
            current_item["match_type"] = "review_needed"
            current_item["review_reason"] = "疑似逗号拆分或去括号后的匹配，需人工审查"
            current_item["review_candidates"] = review_matches

            final_output[out_key] = current_item
            review_count += 1
        else:

            unmatched_output[out_key] = current_item
            unmatched_count += 1

    save_json(final_output, FINAL_OUTPUT_PATH)
    save_json(unmatched_output, UNMATCHED_OUTPUT_PATH)

    print("处理完成：")
    print(f"  直接匹配写入 final_output: {direct_match_count}")
    print(f"  人工审查写入 final_output: {review_count}")
    print(f"  未匹配写入 unmatched_output: {unmatched_count}")
    print()
    print("输出文件：")
    print(f"  {FINAL_OUTPUT_PATH}")
    print(f"  {UNMATCHED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()





import json
import re
from pathlib import Path
from collections import defaultdict




OUTPUT_JSON_PATH = r"output.json"
MSDS_DIR = r"数据集预处理\MSDS"

FINAL_OUTPUT_PATH = r"output_with_reference.json"
UNMATCHED_OUTPUT_PATH = r"unmatched_output.json"





def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s


def remove_bracket_content(s: str) -> str:
    if not s:
        return ""
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\([^()]*\)", "", s)
    return normalize_text(s)


def split_multi_name(s: str):
    if not s:
        return []
    s = s.replace("，", ",").replace("、", ",").replace("/", ",").replace("；", ",")
    parts = [normalize_text(x) for x in s.split(",")]
    return [x for x in parts if x]


def deduplicate_records(records):
    seen = set()
    result = []
    for r in records:
        key = (r["file_name"], r.get("reference", ""))
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result





def build_msds_index(msds_dir):
    exact_index = defaultdict(list)
    relaxed_index = defaultdict(list)

    for fp in Path(msds_dir).glob("*.json"):
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[跳过] 读取失败: {fp} -> {e}")
            continue

        file_stem = fp.stem
        reference = data.get("reference", "")

        sections = data.get("sections", {})
        sec1 = sections.get("第一部分：化学品及企业标识", {})

        title = data.get("标题", "")
        cn_name = sec1.get("中文名称", "")
        cn_alias = sec1.get("中文别名", "")

        record = {
            "file_name": fp.name,
            "reference": reference,
            "标题": title,
            "中文名称": cn_name,
            "中文别名": cn_alias,
        }


        for val in [cn_name, cn_alias]:
            if val and val != "无资料":
                nv = normalize_text(val)
                if nv:
                    exact_index[nv].append(record)


        raw_fields = [file_stem, title, cn_name, cn_alias]

        relaxed_keys = set()

        for raw in raw_fields:
            if not raw or raw == "无资料":
                continue


            nr = normalize_text(raw)
            if nr:
                relaxed_keys.add(nr)


            for part in split_multi_name(raw):
                relaxed_keys.add(part)


            no_bracket = remove_bracket_content(raw)
            if no_bracket:
                relaxed_keys.add(no_bracket)


            cleaned_raw = raw.replace("（", "(").replace("）", ")")
            cleaned_raw = re.sub(r"\([^()]*\)", "", cleaned_raw)
            for part in split_multi_name(cleaned_raw):
                relaxed_keys.add(part)

        for key in relaxed_keys:
            relaxed_index[key].append(record)

    return exact_index, relaxed_index





def main():
    output_data = load_json(OUTPUT_JSON_PATH)
    exact_index, relaxed_index = build_msds_index(MSDS_DIR)

    final_output = {}
    unmatched_output = {}

    matched_count = 0
    unmatched_count = 0
    ambiguous_count = 0

    for out_key, out_info in output_data.items():
        current_item = json.loads(json.dumps(out_info, ensure_ascii=False))

        out_cn_name = out_key
        if isinstance(out_info, dict):
            out_cn_name = out_info.get("中文名称", out_key)

        norm_out_name = normalize_text(out_cn_name)


        exact_matches = deduplicate_records(exact_index.get(norm_out_name, []))

        if len(exact_matches) == 1:
            current_item["reference"] = exact_matches[0]["reference"]
            final_output[out_key] = current_item
            matched_count += 1
            continue

        if len(exact_matches) > 1:

            chosen = next((x for x in exact_matches if x.get("reference")), exact_matches[0])
            current_item["reference"] = chosen["reference"]
            final_output[out_key] = current_item
            matched_count += 1
            ambiguous_count += 1
            continue


        relaxed_matches = deduplicate_records(relaxed_index.get(norm_out_name, []))

        if len(relaxed_matches) == 1:
            current_item["reference"] = relaxed_matches[0]["reference"]
            final_output[out_key] = current_item
            matched_count += 1
            continue

        if len(relaxed_matches) > 1:

            chosen = next((x for x in relaxed_matches if x.get("reference")), relaxed_matches[0])
            current_item["reference"] = chosen["reference"]
            final_output[out_key] = current_item
            matched_count += 1
            ambiguous_count += 1
            continue


        unmatched_output[out_key] = current_item
        unmatched_count += 1

    save_json(final_output, FINAL_OUTPUT_PATH)
    save_json(unmatched_output, UNMATCHED_OUTPUT_PATH)

    print("处理完成：")
    print(f"  已匹配并写入 reference: {matched_count}")
    print(f"  其中多候选但已自动写入: {ambiguous_count}")
    print(f"  未匹配: {unmatched_count}")
    print()
    print("输出文件：")
    print(f"  {FINAL_OUTPUT_PATH}")
    print(f"  {UNMATCHED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()





import json
from pathlib import Path

OUTPUT_PATH = Path("data/output_with_reference.json")
MSDS_DIR = Path("数据集预处理/MSDS")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_cas_number(obj):
    if isinstance(obj, dict):
        if "CAS号" in obj and obj["CAS号"]:
            return str(obj["CAS号"]).strip()
        for value in obj.values():
            cas = find_cas_number(value)
            if cas:
                return cas
    elif isinstance(obj, list):
        for item in obj:
            cas = find_cas_number(item)
            if cas:
                return cas
    return None


def build_reference_to_cas(msds_dir: Path):
    reference_to_cas = {}
    duplicate_references = []

    for json_file in msds_dir.glob("*.json"):
        msds_data = load_json(json_file)
        reference = msds_data.get("reference")
        cas_number = find_cas_number(msds_data)

        if not reference:
            continue

        if reference in reference_to_cas and reference_to_cas[reference] != cas_number:
            duplicate_references.append(reference)
            continue

        reference_to_cas[reference] = cas_number

    return reference_to_cas, duplicate_references


def main():
    output_data = load_json(OUTPUT_PATH)
    reference_to_cas, duplicate_references = build_reference_to_cas(MSDS_DIR)

    updated_count = 0
    missing_reference_count = 0
    missing_cas_count = 0

    for chem_name, chem_info in output_data.items():
        if not isinstance(chem_info, dict):
            continue

        reference = chem_info.get("reference")
        if not reference:
            missing_reference_count += 1
            continue

        cas_number = reference_to_cas.get(reference)
        if cas_number:
            chem_info["CAS号"] = cas_number
            updated_count += 1
        else:
            missing_cas_count += 1

    save_json(output_data, OUTPUT_PATH)

    print("CAS 回填完成")
    print(f"已写入 CAS号: {updated_count}")
    print(f"缺少 reference: {missing_reference_count}")
    print(f"未找到对应 CAS号: {missing_cas_count}")
    if duplicate_references:
        print(f"存在 reference 冲突: {len(duplicate_references)}")


if __name__ == "__main__":
    main()











import json
from pathlib import Path

import pubchempy as pcp
from rdkit import Chem

OUTPUT_PATH = Path("data/output_with_reference.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_valid_value(value):
    return value not in (None, "", "无资料")


def normalize_smiles(smiles: str):
    if not smiles or smiles == "无资料":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def cas_to_smiles(cas_number: str):
    if not cas_number or cas_number == "无资料":
        return None
    try:
        compounds = pcp.get_compounds(cas_number, "name")
    except Exception:
        return None

    for compound in compounds:
        smiles = compound.canonical_smiles or compound.isomeric_smiles
        normalized = normalize_smiles(smiles)
        if normalized:
            return normalized
    return None


def english_name_to_smiles(*names):
    for name in names:
        if not name or name == "无资料":
            continue
        try:
            compounds = pcp.get_compounds(name, "name")
        except Exception:
            continue

        for compound in compounds:
            smiles = compound.canonical_smiles or compound.isomeric_smiles
            normalized = normalize_smiles(smiles)
            if normalized:
                return normalized
    return None


def smiles_to_inchi(smiles: str):
    if not smiles:
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    try:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.InchiToInchiKey(inchi) if inchi else None
    except Exception:
        return None, None

    return inchi, inchikey


def main():
    data = load_json(OUTPUT_PATH)

    updated_count = 0
    from_smiles_count = 0
    from_cas_count = 0
    from_english_count = 0
    failed_count = 0

    for chem_name, chem_info in data.items():
        if not isinstance(chem_info, dict):
            failed_count += 1
            continue

        existing_smiles = chem_info.get("SMILES")
        existing_inchi = chem_info.get("InChI")
        existing_inchikey = chem_info.get("InChIKey")

        smiles = normalize_smiles(existing_smiles)
        source = None

        if smiles:
            source = "SMILES"
        elif not has_valid_value(existing_smiles):
            smiles = cas_to_smiles(chem_info.get("CAS号"))
            if smiles:
                chem_info["SMILES"] = smiles
                source = "CAS号"
            else:
                smiles = english_name_to_smiles(
                    chem_info.get("英文名称"),
                    chem_info.get("英文别名")
                )
                if smiles:
                    chem_info["SMILES"] = smiles
                    source = "英文名称"

        inchi, inchikey = smiles_to_inchi(smiles)
        if inchi and inchikey:
            if not has_valid_value(existing_inchi):
                chem_info["InChI"] = inchi
            if not has_valid_value(existing_inchikey):
                chem_info["InChIKey"] = inchikey
            updated_count += 1
            if source == "SMILES":
                from_smiles_count += 1
            elif source == "CAS号":
                from_cas_count += 1
            elif source == "英文名称":
                from_english_count += 1
        else:
            if not has_valid_value(existing_smiles) and not smiles:
                chem_info["SMILES"] = "无资料"
            if not has_valid_value(existing_inchi):
                chem_info["InChI"] = "无资料"
            if not has_valid_value(existing_inchikey):
                chem_info["InChIKey"] = "无资料"
            failed_count += 1
            print(f"未能生成 InChI/InChIKey: {chem_name}")

    save_json(data, OUTPUT_PATH)

    print("InChI / InChIKey 回填完成")
    print(f"成功写入: {updated_count}")
    print(f"其中直接使用 SMILES: {from_smiles_count}")
    print(f"其中通过 CAS号 补到 SMILES: {from_cas_count}")
    print(f"其中通过英文名称补到 SMILES: {from_english_count}")
    print(f"失败条目: {failed_count}")


if __name__ == "__main__":
    main()





import json
from pathlib import Path

SOURCE_PATH = Path("data/output_with_reference.json")
MSDS_ROOT = Path("数据集预处理/MSDS")
OUTPUT_PATH = Path("output.json")

WANTED_FIELDS = [
    "健康危害", "环境危害", "燃爆危险", "皮肤接触", "眼睛接触", "吸入", "食入", "危险特性", "建规火险分级", "有害燃烧产物",
    "灭火方法", "应急处理", "操作注意事项", "储存注意事项", "监测方法", "工程控制", "呼吸系统防护", "眼睛防护", "身体防护", "手防护",
    "其他防护", "禁配物", "避免接触的条件", "聚合危害", "分解产物", "废弃处置方法", "废弃注意事项", "包装方法", "运输注意事项", "法规信息",
    "生态毒理毒性", "生物降解性", "非生物降解性", "生物富集或生物积累性", "其它有害作用"
]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def extract_msds_fields(msds_obj: dict) -> dict:
    extracted = {field: "无资料" for field in WANTED_FIELDS}
    for section in (msds_obj.get("sections") or {}).values():
        if isinstance(section, dict):
            for field in WANTED_FIELDS:
                if field in section:
                    extracted[field] = section[field]
    return extracted

def merge_msds_fields_into_output() -> None:
    source_data = load_json(SOURCE_PATH)
    reference_to_fields = {}
    for msds_path in MSDS_ROOT.glob("*.json"):
        msds_obj = load_json(msds_path)
        reference = msds_obj.get("reference")
        if reference:
            reference_to_fields[reference] = extract_msds_fields(msds_obj)

    merged_data = {}
    missing_references = []
    for chem_name, chem_info in source_data.items():
        merged_info = dict(chem_info)
        matched_fields = reference_to_fields.get(chem_info.get("reference"))
        if matched_fields is None:
            missing_references.append(chem_name)
            matched_fields = {field: "无资料" for field in WANTED_FIELDS}
        merged_info.update(matched_fields)
        merged_data[chem_name] = merged_info

    OUTPUT_PATH.write_text(json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成: {OUTPUT_PATH}")
    print(f"总条目数: {len(merged_data)}")
    print(f"未匹配 reference 数: {len(missing_references)}")


if __name__ == "__main__":
    merge_msds_fields_into_output()

