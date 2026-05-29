





import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "cluster").exists() and (PROJECT_ROOT.parent.parent / "cluster").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.parent

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL = "gpt-4o"
INPUT_FILE = PROJECT_ROOT / "cluster" / "tail_clusters_all.json"
OUTPUT_FILE = PROJECT_ROOT / "cluster" / "tail_clusters_old_to_new.json"
FAILED_INPUTS_FILE = PROJECT_ROOT / "cluster" / "llm_failed_inputs.jsonl"
TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_API_RETRIES = 4
RETRY_DELAY_SECONDS = 3.0

SYSTEM_PROMPT = """你是一个非常保守的中文术语规范化助手，负责对已经过向量聚类的一组 tail 文本进行二次审查和命名归并。

任务目标：
1. 输入是一组已经初步聚在一起的短文本。
2. 你需要判断哪些文本确实可以归并到同一个规范名称，哪些虽然相似但不应该合并。
3. 输出必须是 old -> new 的映射。

严格规则：
1. 你必须谨慎，宁可少合并，也不要误合并。
2. 如果某个文本与其他文本不能确定是同一语义对象，就让它保持独立，映射为自己。
3. 同一组输入中可以拆成多个子组，不要求全都归到一个名字下。
4. 规范名称必须是中文短语，尽量简洁、稳定、可复用。
5. 规范名称要保留核心语义，不要过度概括。
6. 不能因为化学品名称、法规编号、标准编号、条件短语、动作短语看起来相似，就贸然合并。
7. 如果只是主体化学品不同，而动作或模板句相似，通常不应合并为一个规范名；除非这些文本表达的就是同一个上位规则句式且后续用途明确允许抽象。
8. 涉及法规、标准、编号、化学品名称时，尤其要谨慎。编号不同通常视为不同实体；化学品名称不同通常也不应合并。
9. 如果文本只是轻微表述差异、标点差异、繁简/错别字微差、同义表达，可合并。
10. 输出必须覆盖所有输入项，每个 old 都必须出现一次且仅出现一次。

输出格式要求：
只能输出 JSON，不要输出解释。
格式：
{
  "mapping": {
    "old1": "new1",
    "old2": "new1",
    "old3": "new3"
  }
}
"""


def extract_multi_item_clusters(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    files = data.get("files", [])
    tasks: list[dict] = []

    for file_payload in files:
        file_name = file_payload.get("file")
        clusters = file_payload.get("clusters", [])
        for cluster_index, cluster in enumerate(clusters):
            if isinstance(cluster, list) and len(cluster) > 1:
                tasks.append(
                    {
                        "file": file_name,
                        "cluster_index": cluster_index,
                        "cluster": cluster,
                    }
                )
    return tasks


def build_user_prompt(cluster: list[str]) -> str:
    cluster_lines = "\n".join(f"- {item}" for item in cluster)
    return f"""请对下面这一组 tail 文本做谨慎归并。

要求：
1. 允许拆成多个子组。
2. 如果某项不应与其他项合并，就让它映射到自己。
3. 每个输入项都必须原样作为 mapping 的 key 返回一次，不能遗漏，不能改写 key。
4. new 值是规范名；如果不该合并，就让它映射到自己。
5. 输出只能是 JSON，不要解释。

输入列表：
{cluster_lines}

请返回：
{{
  "mapping": {{
    "原文本1": "规范名1",
    "原文本2": "规范名1",
    "原文本3": "原文本3"
  }}
}}
"""


def build_repair_prompt(
    cluster: list[str],
    partial_mapping: dict[str, str],
    missing_items: list[str],
) -> str:
    cluster_lines = "\n".join(f"- {item}" for item in cluster)
    partial_lines = "\n".join(
        f"- {old} -> {new}" for old, new in partial_mapping.items()
    ) or "- 暂无已确认映射"
    missing_lines = "\n".join(f"- {item}" for item in missing_items)
    canonical_lines = "\n".join(
        f"- {name}" for name in sorted(set(partial_mapping.values()))
    ) or "- 暂无可复用规范名"
    return f"""上一轮结果漏掉了部分输入项，请只补全缺失项。

完整输入列表：
{cluster_lines}

已确认有效的映射：
{partial_lines}

缺失项：
{missing_lines}

可直接复用的规范名：
{canonical_lines}

要求：
1. 只返回缺失项的 mapping。
2. 每个缺失项都必须原样作为 key 返回一次。
3. 如果某个缺失项不应归到已有规范名，就让它映射到自己。
4. 输出只能是 JSON，不要解释。

请返回：
{{
  "mapping": {{
    "缺失项1": "规范名或自己",
    "缺失项2": "规范名或自己"
  }}
}}
"""


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
    return json.loads(content)


def log_failed_input(
    failure_context: dict,
    prompt: str,
    error: Exception,
    attempts: int,
) -> None:
    payload = {
        **failure_context,
        "prompt": prompt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "attempts": attempts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with FAILED_INPUTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_mapping(mapping: dict) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict")

    cleaned: dict[str, str] = {}
    for old, new in mapping.items():
        if not isinstance(old, str):
            raise ValueError("mapping key must be a string")
        if not isinstance(new, str):
            raise ValueError(f"new value must be a string for {old}")
        old = old.strip()
        new = new.strip()
        if not old:
            raise ValueError("mapping key is empty")
        if not new:
            raise ValueError(f"new value is empty for {old}")
        cleaned[old] = new

    return cleaned


def split_mapping_issues(
    expected_items: list[str],
    mapping: dict[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    cleaned = normalize_mapping(mapping)
    expected_set = set(expected_items)
    usable = {old: new for old, new in cleaned.items() if old in expected_set}
    missing = [item for item in expected_items if item not in usable]
    extra = sorted(key for key in cleaned if key not in expected_set)
    return usable, missing, extra


def build_groups(cluster: list[str], mapping: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in cluster:
        name = mapping[item]
        groups.setdefault(name, []).append(item)
    return groups


def merge_groups(existing: dict[str, list[str]], new_groups: dict[str, list[str]]) -> None:
    for name, items in new_groups.items():
        bucket = existing.setdefault(name, [])
        for item in items:
            if item not in bucket:
                bucket.append(item)


def request_mapping(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    failure_context: dict,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json_response(content)
            return {"ok": True, "mapping": normalize_mapping(parsed.get("mapping", {}))}
        except (APIConnectionError, APITimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == MAX_API_RETRIES:
                log_failed_input(failure_context, prompt, exc, attempt)
                return {
                    "ok": False,
                    "attempts": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            print(
                f"Request parsing failed on attempt {attempt}/{MAX_API_RETRIES}: {exc}. "
                f"Retrying in {RETRY_DELAY_SECONDS} seconds..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return {
        "ok": False,
        "attempts": MAX_API_RETRIES,
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error_message": str(last_error) if last_error else "Unknown request failure",
    }


def call_llm(
    client: OpenAI,
    model: str,
    file_name: str,
    cluster_index: int,
    cluster: list[str],
    temperature: float,
) -> dict:
    initial_result = request_mapping(
        client=client,
        model=model,
        prompt=build_user_prompt(cluster),
        temperature=temperature,
        failure_context={
            "file": file_name,
            "cluster_index": cluster_index,
            "stage": "initial_mapping",
            "cluster": cluster,
        },
    )
    if not initial_result["ok"]:
        return {"skipped": True}

    mapping, missing_items, extra_keys = split_mapping_issues(
        cluster,
        initial_result["mapping"],
    )

    warning = {
        "initial_missing_items": list(missing_items),
        "repaired_missing_items": [],
        "fallback_to_self_items": [],
        "ignored_extra_keys": list(extra_keys),
    }

    if missing_items:
        repair_result = request_mapping(
            client=client,
            model=model,
            prompt=build_repair_prompt(cluster, mapping, missing_items),
            temperature=temperature,
            failure_context={
                "file": file_name,
                "cluster_index": cluster_index,
                "stage": "repair_mapping",
                "cluster": cluster,
                "missing_items": missing_items,
            },
        )
        if not repair_result["ok"]:
            return {"skipped": True}

        repaired_mapping, remaining_missing, repair_extra = split_mapping_issues(
            missing_items,
            repair_result["mapping"],
        )
        warning["ignored_extra_keys"].extend(repair_extra)
        warning["repaired_missing_items"] = [
            item for item in missing_items if item in repaired_mapping
        ]
        mapping.update(repaired_mapping)

        for item in remaining_missing:
            mapping[item] = item
        warning["fallback_to_self_items"] = list(remaining_missing)

    groups = build_groups(cluster, mapping)
    return {
        "skipped": False,
        "mapping": mapping,
        "groups": groups,
        "warning": warning,
    }


def main() -> None:
    api_key = API_KEY.strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")

    model = MODEL.strip()
    if not model:
        raise ValueError("Missing MODEL. Please fill in MODEL at the top of this cell.")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL.strip() or None,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    tasks = extract_multi_item_clusters(INPUT_FILE)

    output = {"mapping": {}, "groups": {}, "warnings": []}
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    for index, task in enumerate(tasks, start=1):
        print(
            f"[{index}/{len(tasks)}] processing {task['file']} "
            f"cluster #{task['cluster_index']} ({len(task['cluster'])} items)"
        )
        file_name = task["file"]
        cluster_index = task["cluster_index"]
        result = call_llm(
            client=client,
            model=model,
            file_name=file_name,
            cluster_index=cluster_index,
            cluster=task["cluster"],
            temperature=TEMPERATURE,
        )
        if result["skipped"]:
            print(
                f"Skipped {file_name} cluster #{cluster_index}; details saved to {FAILED_INPUTS_FILE}"
            )
            continue

        output["mapping"].setdefault(file_name, {})
        output["mapping"][file_name].update(result["mapping"])
        file_groups = output["groups"].setdefault(file_name, {})
        merge_groups(file_groups, result["groups"])

        warning = result["warning"]
        if any(warning.values()):
            output["warnings"].append(
                {
                    "file": file_name,
                    "cluster_index": cluster_index,
                    **warning,
                }
            )

        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved progress to {OUTPUT_FILE}")

    print(f"Saved mapping/groups/warnings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()





import json
import os
import re
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "cluster").exists() and (PROJECT_ROOT.parent.parent / "cluster").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.parent

RERUN_API_KEY = os.getenv("OPENAI_API_KEY", "")
RERUN_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
RERUN_MODEL = "gpt-4o"
RERUN_FAILED_INPUTS_FILE = PROJECT_ROOT / "cluster" / "llm_failed_inputs.jsonl"
RERUN_OUTPUT_FILE = PROJECT_ROOT / "cluster" / "tail_clusters_old_to_new_rerun.json"
RERUN_TEMPERATURE = 0.0
RERUN_REQUEST_TIMEOUT_SECONDS = 60.0
RERUN_MAX_API_RETRIES = 4
RERUN_RETRY_DELAY_SECONDS = 3.0

RERUN_SYSTEM_PROMPT = """你是一个非常保守的中文术语规范化助手，负责对已经过向量聚类的一组 tail 文本进行二次审查和命名归并。

任务目标：
1. 输入是一组已经初步聚在一起的短文本。
2. 你需要判断哪些文本确实可以归并到同一个规范名称，哪些虽然相似但不应该合并。
3. 输出必须是 old -> new 的映射。

严格规则：
1. 你必须谨慎，宁可少合并，也不要误合并。
2. 如果某个文本与其他文本不能确定是同一语义对象，就让它保持独立，映射为自己。
3. 同一组输入中可以拆成多个子组，不要求全都归到一个名字下。
4. 规范名称必须是中文短语，尽量简洁、稳定、可复用。
5. 规范名称要保留核心语义，不要过度概括。
6. 不能因为化学品名称、法规编号、标准编号、条件短语、动作短语看起来相似，就贸然合并。
7. 如果只是主体化学品不同，而动作或模板句相似，通常不应合并为一个规范名；除非这些文本表达的就是同一个上位规则句式且后续用途明确允许抽象。
8. 涉及法规、标准、编号、化学品名称时，尤其要谨慎。编号不同通常视为不同实体；化学品名称不同通常也不应合并。
9. 如果文本只是轻微表述差异、标点差异、繁简/错别字微差、同义表达，可合并。
10. 输出必须覆盖所有输入项，每个 old 都必须出现一次且仅出现一次。

输出格式要求：
只能输出 JSON，不要输出解释。
格式：
{
  "mapping": {
    "old1": "new1",
    "old2": "new1",
    "old3": "new3"
  }
}
"""


def split_large_cluster(cluster: list[str]) -> list[list[str]]:
    if not cluster:
        return []
    target_chunks = 4 if len(cluster) < 300 else 5
    chunk_size = (len(cluster) + target_chunks - 1) // target_chunks
    return [cluster[i : i + chunk_size] for i in range(0, len(cluster), chunk_size)]


def split_chunk_in_half(cluster: list[str]) -> list[list[str]]:
    if len(cluster) <= 1:
        return [cluster]
    midpoint = len(cluster) // 2
    return [cluster[:midpoint], cluster[midpoint:]]


def build_user_prompt(cluster: list[str]) -> str:
    cluster_lines = "\n".join(f"- {item}" for item in cluster)
    return f"""请对下面这一组 tail 文本做谨慎归并。

要求：
1. 允许拆成多个子组。
2. 如果某项不应与其他项合并，就让它映射到自己。
3. 每个输入项都必须原样作为 mapping 的 key 返回一次，不能遗漏，不能改写 key。
4. new 值是规范名；如果不该合并，就让它映射到自己。
5. 输出只能是 JSON，不要解释。

输入列表：
{cluster_lines}

请返回：
{{
  "mapping": {{
    "原文本1": "规范名1",
    "原文本2": "规范名1",
    "原文本3": "原文本3"
  }}
}}
"""


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
    return json.loads(content)


def normalize_mapping(mapping: dict) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dict")

    cleaned: dict[str, str] = {}
    for old, new in mapping.items():
        if not isinstance(old, str):
            raise ValueError("mapping key must be a string")
        if not isinstance(new, str):
            raise ValueError(f"new value must be a string for {old}")
        old = old.strip()
        new = new.strip()
        if not old:
            raise ValueError("mapping key is empty")
        if not new:
            raise ValueError(f"new value is empty for {old}")
        cleaned[old] = new

    return cleaned


def build_groups(cluster: list[str], mapping: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in cluster:
        name = mapping[item]
        groups.setdefault(name, []).append(item)
    return groups


def merge_groups(existing: dict[str, list[str]], new_groups: dict[str, list[str]]) -> None:
    for name, items in new_groups.items():
        bucket = existing.setdefault(name, [])
        for item in items:
            if item not in bucket:
                bucket.append(item)


def request_mapping(client: OpenAI, prompt: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, RERUN_MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=RERUN_MODEL,
                temperature=RERUN_TEMPERATURE,
                messages=[
                    {"role": "system", "content": RERUN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=RERUN_REQUEST_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json_response(content)
            return {"ok": True, "mapping": normalize_mapping(parsed.get("mapping", {}))}
        except (APIConnectionError, APITimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == RERUN_MAX_API_RETRIES:
                return {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            print(
                f"Chunk request failed on attempt {attempt}/{RERUN_MAX_API_RETRIES}: {exc}. "
                f"Retrying in {RERUN_RETRY_DELAY_SECONDS} seconds..."
            )
            time.sleep(RERUN_RETRY_DELAY_SECONDS)

    return {
        "ok": False,
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error_message": str(last_error) if last_error else "Unknown request failure",
    }


def call_llm_for_chunk(client: OpenAI, cluster: list[str]) -> dict:
    result = request_mapping(client, build_user_prompt(cluster))
    if not result["ok"]:
        return {"skipped": True, **result}

    mapping = result["mapping"]
    missing = sorted(set(cluster) - set(mapping))
    extra = sorted(set(mapping) - set(cluster))
    if missing or extra:
        return {
            "skipped": True,
            "error_type": "ValueError",
            "error_message": (
                "chunk mapping keys do not cover all cluster items, "
                f"missing={missing[:5]}, extra={extra[:5]}"
            ),
        }

    return {
        "skipped": False,
        "mapping": mapping,
        "groups": build_groups(cluster, mapping),
    }


def recover_chunk_recursively(
    client: OpenAI,
    cluster: list[str],
    label: str,
    depth: int = 0,
) -> dict:
    result = call_llm_for_chunk(client, cluster)
    if not result["skipped"]:
        return result

    if len(cluster) <= 1:
        return result

    indent = "  " * (depth + 1)
    error_message = result.get("error_message", "Unknown chunk failure")
    print(f"{indent}refining {label} into 2 smaller pieces because: {error_message}")

    merged_mapping: dict[str, str] = {}
    merged_groups: dict[str, list[str]] = {}
    for sub_index, subcluster in enumerate(split_chunk_in_half(cluster), start=1):
        print(f"{indent}subchunk {sub_index}/2 ({len(subcluster)} items)")
        child_result = recover_chunk_recursively(
            client=client,
            cluster=subcluster,
            label=f"{label}.{sub_index}",
            depth=depth + 1,
        )
        if child_result["skipped"]:
            return child_result
        merged_mapping.update(child_result["mapping"])
        merge_groups(merged_groups, child_result["groups"])

    return {
        "skipped": False,
        "mapping": merged_mapping,
        "groups": merged_groups,
    }


def rerun_failed_large_clusters() -> None:
    if not RERUN_FAILED_INPUTS_FILE.exists():
        print(f"No failed input file found: {RERUN_FAILED_INPUTS_FILE}")
        return

    if not RERUN_API_KEY.strip():
        raise ValueError("Missing OPENAI_API_KEY.")
    if not RERUN_MODEL.strip():
        raise ValueError("Missing RERUN_MODEL. Please fill it in at the top of this cell.")

    records = [
        json.loads(line)
        for line in RERUN_FAILED_INPUTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        print(f"No failed records found in {RERUN_FAILED_INPUTS_FILE}")
        return

    output = {"mapping": {}, "groups": {}, "warnings": []}
    client = OpenAI(
        api_key=RERUN_API_KEY,
        base_url=RERUN_BASE_URL.strip() or None,
        timeout=RERUN_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    recovered = 0
    for record in records:
        file_name = record.get("file", "")
        cluster_index = record.get("cluster_index", -1)
        cluster = record.get("cluster", [])
        if not file_name or not isinstance(cluster, list) or not cluster:
            print(f"Skipped malformed failed record: {record}")
            continue

        subclusters = split_large_cluster(cluster)
        print(
            f"Rerunning {file_name} cluster #{cluster_index} as {len(subclusters)} smaller clusters"
        )

        merged_mapping: dict[str, str] = {}
        merged_groups: dict[str, list[str]] = {}
        chunk_failed = False

        for part_index, subcluster in enumerate(subclusters, start=1):
            print(f"  chunk {part_index}/{len(subclusters)} ({len(subcluster)} items)")
            result = recover_chunk_recursively(
                client=client,
                cluster=subcluster,
                label=f"chunk {part_index}/{len(subclusters)}",
            )
            if result["skipped"]:
                output["warnings"].append(
                    {
                        "file": file_name,
                        "cluster_index": cluster_index,
                        "chunk_index": part_index,
                        "chunk_size": len(subcluster),
                        "error_type": result.get("error_type", "UnknownError"),
                        "error_message": result.get("error_message", "Unknown chunk failure"),
                    }
                )
                print(
                    f"  chunk {part_index} failed again: {result.get('error_message', 'Unknown chunk failure')}"
                )
                chunk_failed = True
                break

            merged_mapping.update(result["mapping"])
            merge_groups(merged_groups, result["groups"])

        if chunk_failed:
            continue

        if set(merged_mapping) != set(cluster):
            output["warnings"].append(
                {
                    "file": file_name,
                    "cluster_index": cluster_index,
                    "error_type": "ValueError",
                    "error_message": "chunked rerun produced incomplete mapping",
                }
            )
            print(
                f"  chunked rerun produced incomplete mapping for {file_name} cluster #{cluster_index}"
            )
            continue

        output["mapping"].setdefault(file_name, {}).update(merged_mapping)
        file_groups = output["groups"].setdefault(file_name, {})
        merge_groups(file_groups, merged_groups)
        recovered += 1
        print(f"Recovered {file_name} cluster #{cluster_index}")

    with RERUN_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Recovered {recovered} failed clusters -> saved to {RERUN_OUTPUT_FILE}")




