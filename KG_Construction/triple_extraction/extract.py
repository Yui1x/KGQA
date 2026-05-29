





import json
import os
import time
from pathlib import Path

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "output.json").exists() and (PROJECT_ROOT.parent.parent / "output.json").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.parent

INPUT_PATH = PROJECT_ROOT / "output.json"
OUTPUT_DIR = PROJECT_ROOT / "data2"
OUTPUT_PATH = OUTPUT_DIR / "chemical_entity_relation_extraction.json"
PROGRESS_PATH = OUTPUT_DIR / "chemical_entity_relation_progress.json"
MAX_ITEMS = 1
SLEEP_SECONDS = 0.2

EXCLUDED_INPUT_FIELDS = {
    "中文名称",
    "中文别名",
    "英文名称",
    "英文别名",
    "SMILES",
    "reference",
    "CAS号",
    "InChI",
    "InChIKey",
}

NO_DATA_VALUES = {
    "",
    "无资料",
    "无资料。",
    "暂无资料",
    "暂无资料。",
    "暂无",
    "暂无。",
    "无",
    "无。",
    "未见资料",
    "未见资料。",
    "未提及",
    "未提及。",
    "none",
    "null",
}

ALLOWED_RELATIONS = [
    "具有健康危害",
    "具有环境危害",
    "具有燃爆风险",
    "皮肤接触后的应急处置",
    "眼部接触后的应急处置",
    "吸入后的应急处置",
    "误食后的应急处置",
    "具有危险反应特征",
    "属于某一火灾危险类别",
    "燃烧时会产生某些有害产物",
    "适用的灭火方式",
    "泄漏或事故时的应急处理方式",
    "安全操作要求",
    "存储条件",
    "存储要求",
    "推荐监测或检测方法",
    "工程控制措施",
    "呼吸防护要求",
    "眼面部防护要求",
    "身体防护要求",
    "手部防护要求",
    "其他个人防护与卫生要求",
    "禁配物",
    "应避免的接触或环境条件",
    "是否存在聚合风险",
    "废弃处理建议",
    "包装要求",
    "运输要求",
    "适用法规或监管分类",
    "其他有害影响",
]

FIELD_TO_RELATION = {
    "健康危害": "具有健康危害",
    "环境危害": "具有环境危害",
    "燃爆危险": "具有燃爆风险",
    "皮肤接触": "皮肤接触后的应急处置",
    "眼睛接触": "眼部接触后的应急处置",
    "吸入": "吸入后的应急处置",
    "食入": "误食后的应急处置",
    "危险特性": "具有危险反应特征",
    "建规火险分级": "属于某一火灾危险类别",
    "有害燃烧产物": "燃烧时会产生某些有害产物",
    "灭火方法": "适用的灭火方式",
    "隔离应急处理": "泄漏或事故时的应急处理方式",
    "操作注意事项": "安全操作要求",
    "储存注意事项": ["存储条件", "存储要求"],
    "监测方法": "推荐监测或检测方法",
    "工程控制": "工程控制措施",
    "呼吸系统防护": "呼吸防护要求",
    "眼睛防护": "眼面部防护要求",
    "身体防护": "身体防护要求",
    "手防护": "手部防护要求",
    "其他防护": "其他个人防护与卫生要求",
    "禁配物": "禁配物",
    "避免接触的条件": "应避免的接触或环境条件",
    "聚合危害": "是否存在聚合风险",
    "废弃处置方法": "废弃处理建议",
    "包装方法": "包装要求",
    "运输注意事项": "运输要求",
    "法规信息": "适用法规或监管分类",
    "其它有害作用": "其他有害影响",
}

DIRECT_ONLY_FIELDS = {"??????"}
LLM_TARGET_FIELDS = [
    field for field in FIELD_TO_RELATION.keys() if field not in DIRECT_ONLY_FIELDS
]

RELATION_GUIDANCE = "\n".join(
    f"- {relation}" for relation in ALLOWED_RELATIONS
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

SYSTEM_PROMPT = f"""
你是一名危险化学品知识图谱信息抽取助手。
任务是严格依据给定字段内容，抽取结构化三元信息。
不要摘要，不要改写，不要补全常识，不要引入字段外信息，不要编造。

【抽取范围】
只抽取输入中给出的字段内容。
每条三元组的 relation 只能从下列关系列表中选择，不能自造、不能改写、不能合并、不能近义替换：
{RELATION_GUIDANCE}

【字段与关系约束】
- 健康危害 -> 具有健康危害
- 环境危害 -> 具有环境危害
- 燃爆危险 -> 具有燃爆风险
- 皮肤接触 -> 皮肤接触后的应急处置
- 眼睛接触 -> 眼部接触后的应急处置
- 吸入 -> 吸入后的应急处置
- 食入 -> 误食后的应急处置
- 危险特性 -> 具有危险反应特征
- 建规火险分级 -> 属于某一火灾危险类别
- 有害燃烧产物 -> 燃烧时会产生某些有害产物
- 灭火方法 -> 适用的灭火方式
- 隔离应急处理 -> 泄漏或事故时的应急处理方式
- 操作注意事项 -> 安全操作要求
- 储存注意事项 -> 仅可抽为“存储条件”或“存储要求”
- 监测方法 -> 推荐监测或检测方法
- 工程控制 -> 工程控制措施
- 呼吸系统防护 -> 呼吸防护要求
- 眼睛防护 -> 眼面部防护要求
- 身体防护 -> 身体防护要求
- 手防护 -> 手部防护要求
- 其他防护 -> 其他个人防护与卫生要求
- 禁配物 -> 禁配物
- 避免接触的条件 -> 应避免的接触或环境条件
- 聚合危害 -> 是否存在聚合风险
- 废弃处置方法 -> 废弃处理建议
- 包装方法 -> 包装要求
- 运输注意事项 -> 运输要求
- 法规信息 -> 适用法规或监管分类
- 其它有害作用 -> 其他有害影响

【特别要求】
- 如果字段内容与允许关系都不匹配，则不抽取。
- 如果字段内容为“无资料”或缺乏明确可抽取信息，则不抽取。
- “储存注意事项”中，环境、场所、温度、光照、密封、通风等偏条件类内容抽为“存储条件”；分开存放、禁忌混储、搬运/管理要求、设施配置等偏规范要求抽为“存储要求”。
- 列举内容必须拆分成多条。
- 相同的 head-relation-tail 不要重复。
- evidence 必须直接来自原文。

【输出要求】
输出必须是一个 JSON 对象，且仅包含一个键：triples。
triples 是数组；数组中每个元素只包含以下 4 个键：
head
relation
tail
evidence

不要输出任何说明、Markdown、代码块或其他键。
""".strip()


def is_no_data(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() in NO_DATA_VALUES


def filter_target_fields(item: dict) -> dict:
    filtered = {}
    for field in LLM_TARGET_FIELDS:
        if field in EXCLUDED_INPUT_FIELDS or field in DIRECT_ONLY_FIELDS:
            continue
        value = item.get(field, "")
        if is_no_data(value):
            continue
        filtered[field] = str(value).strip()
    return filtered


def merge_triples(*triple_groups: list[dict]) -> list[dict]:
    merged = []
    seen = set()

    for group in triple_groups:
        for triple in group:
            key = (
                str(triple.get("head", "")).strip(),
                str(triple.get("relation", "")).strip(),
                str(triple.get("tail", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(triple)

    return merged


def build_user_prompt(chemical_name: str, item: dict) -> str:
    payload = {
        "chemical_name": chemical_name,
        "allowed_relations": ALLOWED_RELATIONS,
        "field_relation_mapping": FIELD_TO_RELATION,
        "target_fields": filter_target_fields(item),
    }
    return (
        "请对下面这一个化学品样本执行实体关系抽取。\n"
        "请严格只从给定的 allowed_relations 中选择 relation；没有匹配关系时直接跳过。\n"
        "请不要使用被舍弃字段，不要补全常识。\n\n"
        f"输入数据：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_direct_triples(chemical_name: str, item: dict) -> list[dict]:
    reference = str(item.get("reference", "")).strip()
    fire_level = str(item.get("建规火险分级", "")).strip()

    if is_no_data(fire_level):
        return []

    return [
        {
            "head": chemical_name,
            "relation": "属于某一火灾危险类别",
            "tail": fire_level,
            "evidence": fire_level,
            "reference": reference,
        }
    ]


def normalize_result(chemical_name: str, reference: str, result: dict) -> list[dict]:
    triples = result.get("triples", []) if isinstance(result, dict) else []
    normalized = []
    seen = set()

    for triple in triples:
        if not isinstance(triple, dict):
            continue

        head = str(triple.get("head", chemical_name)).strip() or chemical_name
        relation = str(triple.get("relation", "")).strip()
        tail = str(triple.get("tail", "")).strip()
        evidence = str(triple.get("evidence", "")).strip()

        if relation not in ALLOWED_RELATIONS or not tail:
            continue
        if is_no_data(tail) or is_no_data(evidence):
            continue

        key = (head, relation, tail)
        if key in seen:
            continue
        seen.add(key)

        normalized.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "evidence": evidence,
                "reference": reference,
            }
        )

    return normalized


def extract_relations_for_chemical(chemical_name: str, item: dict) -> list[dict]:
    reference = str(item.get("reference", "")).strip()
    direct_triples = build_direct_triples(chemical_name, item)
    filtered_fields = filter_target_fields(item)

    if not filtered_fields:
        return direct_triples

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(chemical_name, item)},
        ],
    )

    content = response.choices[0].message.content
    result = json.loads(content)
    llm_triples = normalize_result(chemical_name, reference, result)
    return merge_triples(direct_triples, llm_triples)


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize_progress(all_names: list[str]) -> dict:
    if PROGRESS_PATH.exists():
        progress = load_json_file(
            PROGRESS_PATH,
            {"pending_chemicals": [], "failed_chemicals": {}},
        )
        progress.setdefault("pending_chemicals", [])
        progress.setdefault("failed_chemicals", {})
        return progress

    pending = all_names[:MAX_ITEMS] if MAX_ITEMS is not None else all_names[:]
    progress = {
        "pending_chemicals": pending,
        "failed_chemicals": {},
    }
    save_json_file(PROGRESS_PATH, progress)
    return progress


def run_extraction():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    all_names = list(data.keys())
    progress = initialize_progress(all_names)
    pending_names = progress["pending_chemicals"]
    items = [(name, data[name]) for name in pending_names if name in data]
    all_triples = load_json_file(OUTPUT_PATH, [])

    if not isinstance(all_triples, list):
        raise ValueError(f"{OUTPUT_PATH} 必须是 JSON 数组格式")

    if not items:
        print("没有待处理的化学品，程序结束。")
    else:
        print(f"待处理化学品数量: {len(items)}")

    for idx, (chemical_name, item) in enumerate(items, start=1):
        try:
            triples = extract_relations_for_chemical(chemical_name, item)
            all_triples.extend(triples)
            progress["pending_chemicals"] = [
                name for name in progress["pending_chemicals"] if name != chemical_name
            ]
            progress["failed_chemicals"].pop(chemical_name, None)
            save_json_file(OUTPUT_PATH, all_triples)
            save_json_file(PROGRESS_PATH, progress)
            print(f"[{idx}/{len(items)}] {chemical_name} -> {len(triples)} triples")
        except Exception as exc:
            progress["failed_chemicals"][chemical_name] = str(exc)
            save_json_file(PROGRESS_PATH, progress)
            print(f"[{idx}/{len(items)}] {chemical_name} -> ERROR: {exc}")
        time.sleep(SLEEP_SECONDS)

    print(f"\n抽取完成，结果已保存到: {OUTPUT_PATH}")
    print(f"总三元组条数: {len(all_triples)}")
    print(f"剩余待处理条数: {len(progress['pending_chemicals'])}")
    if progress["failed_chemicals"]:
        print(f"失败条数: {len(progress['failed_chemicals'])}")
        print("示例错误:", next(iter(progress["failed_chemicals"].items())))
if __name__ == "__main__":
    run_extraction()





import json
import os
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data_1").exists() and (PROJECT_ROOT.parent.parent / "data_1").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

if "load_json_file" not in globals():
    def load_json_file(path: Path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

if "save_json_file" not in globals():
    def save_json_file(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

REACTION_TYPE_INPUT_PATH = PROJECT_ROOT / "data_1" / "ord_reaction_pairs_with_notes.json"
REACTION_TYPE_OUTPUT_PATH = PROJECT_ROOT / "data_1" / "ord_reaction_pairs_with_notes_llm_relation.json"
REACTION_TYPE_PROGRESS_PATH = PROJECT_ROOT / "data_1" / "ord_reaction_pairs_with_notes_llm_relation_progress.json"
REACTION_TYPE_MAX_ITEMS = None
REACTION_TYPE_SLEEP_SECONDS = 0.2
UNKNOWN_REACTION_TYPE = "未知"

REACTION_TYPE_SYSTEM_PROMPT = "\n".join([
    "你是一名化学反应类型判定助手。",
    "你的任务不是总结实验步骤，不是提取全部信息，也不是解释机理，而是只根据输入内容判断该反应最核心的反应方式。",
    "",
    "【任务要求】",
    "1. 若输入描述的是单步反应，只输出一个最核心的反应方式标签。",
    "2. 若输入明确是多步反应、合成路线、分步工艺或含有多个 step/stage/阶段/步骤，则按反应发生顺序输出多个反应方式标签，并且每个标签之间必须使用 -> 连接。",
    "   例如：还原胺化->水解",
    "   不要使用逗号、顿号、斜杠、分号或换行。",
    "3. 判定时应优先输出最具体、最合适的反应类型；若无法可靠判断到具体命名反应或细分类，必须退而输出更上位的反应方式，而不是轻易输出未知。",
    "4. 若输入是多步路线，但无法可靠判断全部步骤，则只输出能够明确判断的步骤标签；若连一个步骤也无法明确判断，则输出未知。",
    "5. 如果无法明确拆分每一步，但能判断总体核心反应方式，则只输出一个上位标签。",
    "6. 不要输出步骤、条件、产率、后处理、表征信息。",
    "7. 必须严格依据输入中明确出现的反应物、产物、关键试剂、催化体系、显式反应术语或结构变化来判断。",
    f"8. 只有当连上位反应方式也无法从文本中可靠判断时，才输出“{UNKNOWN_REACTION_TYPE}”。",
    "9. 不允许根据常识臆测，不允许编造。",
    "10. 若输入描述的是具体命名反应，优先输出该命名反应；若无法确定具体命名反应，则输出较上位的反应方式。",
    "11. 若正文中出现明确反应术语（如 hydrogenation、cyclisation、deprotection、aromatisation、disproportionation、metathesis、polymerization、copolymerization、grafting），优先依据这些显式术语判断。",
    "12. 若 head 和 tail 与正文证据明显不一致，优先依据正文。",
    "13. 若仅出现 amine 与 aldehyde/ketone 反应，但未出现还原剂，不要判为还原胺化；只能输出缩合、胺与醛/酮反应，或未知。",
    "14. 若文本主要是工艺或催化过程描述（如 reactor、guard column、purge、feed、conversion、selectivity），仍需优先识别其中明确出现的反应类型，不要因为工艺细节过多而忽略显式反应术语。",
    "",
    "【可输出标签示例（不限于以下）】",
    "第一层：具体命名反应或高特异类型",
    "还原胺化",
    "Buchwald-Hartwig 胺化",
    "Suzuki 偶联",
    "Heck 反应",
    "Sonogashira 偶联",
    "Friedel-Crafts 烷基化",
    "Friedel-Crafts 酰基化",
    "Michael 加成",
    "Diels-Alder 反应",
    "硝基还原",
    "氧化芳构化",
    "烯烃复分解",
    "烯烃复分解歧化反应",
    "Boc 脱保护",
    "酸水解脱保护",
    "环氧开环",
    "",
    "第二层：中等粒度反应类型",
    "偶联",
    "胺化",
    "酰化",
    "酰胺化",
    "酯化",
    "醚化",
    "N-烷基化",
    "O-烷基化",
    "亲核取代",
    "亲电取代",
    "卤代",
    "脱卤",
    "加成",
    "消除",
    "缩合",
    "环化",
    "开环",
    "保护",
    "脱保护",
    "水解",
    "氧化",
    "还原",
    "氢化还原",
    "异构化",
    "重排",
    "歧化反应",
    "聚合",
    "共聚",
    "接枝改性",
    "自由基聚合",
    "开环聚合",
    "聚合改性",
    "盐形成",
    "中和",
    "",
    "第三层：上位反应方式",
    "官能团转化",
    "C-C 键形成",
    "C-N 键形成",
    "C-O 键形成",
    "取代反应",
    "加成反应",
    "消除反应",
    "氧化反应",
    "还原反应",
    "聚合反应",
    UNKNOWN_REACTION_TYPE,
    "",
    "【判定优先级】",
    "1. 正文中显式给出的反应名称或术语。",
    "2. 关键试剂、催化剂、底物组合能唯一指向的经典反应类型。",
    "3. 由明确结构变化可判断出的中等粒度反应类型。",
    "4. 若仍无法细分，则输出更上位的反应方式。",
    "5. 若以上都不足，则输出未知。",
    "",
    "【降级判定原则】",
    "- 能判定具体命名反应时，输出具体命名反应。",
    "- 不能判定命名反应，但能判定到中等粒度类型时，输出中等粒度类型。",
    "- 不能判定中等粒度类型，但能判定为上位反应方式时，输出上位反应方式。",
    "- 不要因为无法确定机理、立体选择性、区域选择性、催化循环，就直接输出未知。",
    "",
    "【判定规则】",
    "- 胺 + 醛/酮 + 还原剂（如 NaBH(OAc)3、NaBH3CN、STAB） -> 还原胺化。",
    "- 芳基卤化物 + 胺 + Pd/配体/碱 -> Buchwald-Hartwig 胺化；若不能完全确定，可输出胺化或偶联。",
    "- 硼酸/硼酸酯 + 芳基卤化物 + Pd/碱 -> Suzuki 偶联；若不能完全确定，可输出偶联。",
    "- 酸 + 醇生成酯，或酸衍生物与醇反应生成酯 -> 酯化。",
    "- 胺与酸/酰氯/活化酯生成酰胺 -> 酰胺化。",
    "- 酸衍生物引入酰基 -> 酰化。",
    "- 卤代烃、磺酸酯、甲磺酸酯等与胺/醇/其他亲核试剂发生取代 -> 亲核取代；若明确是胺引入烷基链，可输出 N-烷基化；若明确是醇或酚氧负离子引入烷基，可输出 O-烷基化。",
    "- 硝基变氨基 -> 硝基还原。",
    "- 出现 hydrogenation、H2/Pd、Pd/C 加氢等 -> 氢化还原；若无法更细分，可输出还原。",
    "- 出现 cyclisation、cyclized、ring closure 等 -> 环化。",
    "- 出现 ring opening、epoxide opening、开环等 -> 开环；若为单体聚合开环，则可输出开环聚合。",
    "- 出现 deprotection、deprotected、Boc removal、Cbz removal、acid hydrolysis removing protecting group 等 -> 脱保护；若明确是酸水解脱保护，可输出酸水解脱保护。",
    "- 出现 protection、protected、Boc protection、silylation 等 -> 保护。",
    "- 出现 hydrolysis、saponification 等 -> 水解；若只是脱保护的一部分且保护基信息明确，优先输出脱保护。",
    "- 出现 aromatisation、aromatization，或 DDQ 等脱氢芳构化体系 -> 氧化芳构化。",
    "- 出现 oxidation、oxidized、氧化剂体系且底物被氧化 -> 氧化；若无法细分，可输出氧化反应。",
    "- 出现 reduction、reduced、还原剂体系且底物被还原 -> 还原；若无法细分，可输出还原反应。",
    "- 出现 halogenation、chlorination、bromination、iodination -> 卤代。",
    "- 出现 dehalogenation、hydrodehalogenation -> 脱卤。",
    "- 出现 elimination、dehydration、dehydrohalogenation -> 消除。",
    "- 出现 condensation，或明确脱去小分子形成新键 -> 缩合。",
    "- 出现 rearrangement、rearranged -> 重排。",
    "- 出现 isomerization、isomerized -> 异构化。",
    "- 出现 disproportionation -> 歧化反应。",
    "- 出现 metathesis，且底物/产物为烯烃转化 -> 烯烃复分解；若同时明确出现 disproportionation，可输出烯烃复分解歧化反应。",
    "- 出现 polymerization、polymerized -> 聚合。",
    "- 出现 copolymer、copolymerization -> 共聚。",
    "- 出现 graft、grafting、graft polymer、接枝到聚合物骨架 -> 接枝改性。",
    "- 若文本明确提到 polymer modifier、resin modifier、modified polyolefin、copolymer、graft polymer 等，并有单体参与形成聚合物或改性树脂，应优先判为聚合、共聚、接枝改性或聚合改性；无法细分时，不要输出未知，可输出聚合改性或共聚。",
    "- 若涉及 maleic anhydride 与烯烃、聚烯烃等形成改性树脂、共聚物、接枝共聚物或树脂改性剂，可输出共聚、接枝改性或聚合改性；若机理不清但 copolymer 明确，优先输出共聚。",
    "- 若仅能看出形成 C-C 键或引入烷基，但无法确定更具体类型 -> 烷基化、加成、偶联或 C-C 键形成中更合适者。",
    "- 若仅能看出形成 C-N 键，但无法确定更具体类型 -> 胺化或 C-N 键形成。",
    "- 若仅能看出形成 C-O 键，但无法确定更具体类型 -> 醚化、酯化或 C-O 键形成中更合适者。",
    "- 仅出现 amine 与 aldehyde/ketone 反应，但未见还原剂 -> 缩合 或 未知，不得输出还原胺化。",
    f"- 若只能看出发生了还原，但不能确定更具体类型，可输出‘还原’或‘氢化还原’；若连上位还原类型都无法支持，再输出‘{UNKNOWN_REACTION_TYPE}’。",
    "",
    "【多步反应判定规则】",
    "- 若文本中包含 step (a)、step (b)、first、then、followed by、subsequently、in step (x) 等明显多步信号，应优先按步骤分别判定。",
    "- 多步输出时，标签顺序必须与文本中的反应顺序一致。",
    "- 不要把纯操作步骤（purge、heat、cool、wash、dry、concentrate、purify、analyze）当作反应方式。",
    "- 若某一步只是后处理或纯化，不输出该步标签。",
    "",
    "【输出格式】",
    "只输出一个 JSON 对象：",
    "{",
    '  "reaction_type": "反应方式标签；若多步则用->连接"',
    "}",
])


def build_reaction_record_key(item: dict) -> str:
    return "||".join(
        str(item.get(field, "")).strip()
        for field in ("head", "tail", "evidence", "reference")
    )


def extract_reaction_text(item: dict) -> str:
    relation = item.get("relation")
    if isinstance(relation, str):
        return relation.strip()
    if isinstance(relation, dict):
        procedure_details = relation.get("procedure_details")
        if isinstance(procedure_details, str) and procedure_details.strip():
            return procedure_details.strip()

        parts = []
        for key, value in relation.items():
            if isinstance(value, str) and value.strip():
                parts.append(f"{key}: {value.strip()}")
            elif isinstance(value, (dict, list)) and value:
                parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        return "\n".join(parts).strip()
    return ""


def build_reaction_type_user_prompt(item: dict) -> str:
    reaction_text = extract_reaction_text(item)
    payload = {
        "head": str(item.get("head", "")).strip(),
        "tail": str(item.get("tail", "")).strip(),
        "reaction_text": reaction_text,
    }
    return "【输入】\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def classify_reaction_type(item: dict) -> dict:
    reaction_text = extract_reaction_text(item)
    if not reaction_text:
        return {"reaction_type": UNKNOWN_REACTION_TYPE}

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REACTION_TYPE_SYSTEM_PROMPT},
            {"role": "user", "content": build_reaction_type_user_prompt(item)},
        ],
    )

    content = response.choices[0].message.content
    result = json.loads(content)
    reaction_type = str(result.get("reaction_type", UNKNOWN_REACTION_TYPE)).strip() or UNKNOWN_REACTION_TYPE
    return {"reaction_type": reaction_type}


def should_keep_reaction_type_result(result: dict) -> bool:
    reaction_type = str(result.get("reaction_type", "")).strip()
    return bool(reaction_type) and reaction_type not in {UNKNOWN_REACTION_TYPE, "??"}


def normalize_reaction_type_record(item: dict, result: dict) -> dict:
    return {
        "head": str(item.get("head", "")).strip(),
        "relation": str(result.get("reaction_type", UNKNOWN_REACTION_TYPE)).strip() or UNKNOWN_REACTION_TYPE,
        "tail": str(item.get("tail", "")).strip(),
        "evidence": str(item.get("evidence", "")).strip(),
        "reference": str(item.get("reference", "")).strip(),
    }


def initialize_reaction_type_progress(records: list[dict], completed_keys: set[str]) -> dict:
    if REACTION_TYPE_PROGRESS_PATH.exists():
        progress = load_json_file(
            REACTION_TYPE_PROGRESS_PATH,
            {"pending_records": [], "failed_records": {}},
        )
        progress.setdefault("pending_records", [])
        progress.setdefault("failed_records", {})
        if progress["pending_records"]:
            return progress

    all_keys = [build_reaction_record_key(item) for item in records]
    pending_keys = [key for key in all_keys if key not in completed_keys]
    if REACTION_TYPE_MAX_ITEMS is not None:
        pending_keys = pending_keys[:REACTION_TYPE_MAX_ITEMS]

    progress = {
        "pending_records": pending_keys,
        "failed_records": {},
    }
    save_json_file(REACTION_TYPE_PROGRESS_PATH, progress)
    return progress


def run_reaction_type_extraction():
    records = load_json_file(REACTION_TYPE_INPUT_PATH, [])
    if not isinstance(records, list):
        raise ValueError(f"{REACTION_TYPE_INPUT_PATH} must be a JSON array")

    existing_results = load_json_file(REACTION_TYPE_OUTPUT_PATH, [])
    if not isinstance(existing_results, list):
        raise ValueError(f"{REACTION_TYPE_OUTPUT_PATH} must be a JSON array")

    result_by_key = {
        build_reaction_record_key(item): item
        for item in existing_results
        if isinstance(item, dict)
    }
    progress = initialize_reaction_type_progress(records, set(result_by_key))
    record_by_key = {
        build_reaction_record_key(item): item
        for item in records
        if isinstance(item, dict)
    }
    pending_keys = [
        key
        for key in progress["pending_records"]
        if key in record_by_key and key not in result_by_key
    ]

    if not pending_keys:
        print("没有待处理的 ORD 反应记录")
        return

    print(f"待处理 ORD 记录数: {len(pending_keys)}")
    for idx, key in enumerate(pending_keys, start=1):
        item = record_by_key[key]
        try:
            result = classify_reaction_type(item)
            if should_keep_reaction_type_result(result):
                result_by_key[key] = normalize_reaction_type_record(item, result)
                progress["pending_records"] = [k for k in progress["pending_records"] if k != key]
                progress["failed_records"].pop(key, None)
                ordered_results = [result_by_key[k] for k in sorted(result_by_key)]
                save_json_file(REACTION_TYPE_OUTPUT_PATH, ordered_results)
                save_json_file(REACTION_TYPE_PROGRESS_PATH, progress)
                print(
                    f"[{idx}/{len(pending_keys)}] {item.get('head')} -> {item.get('tail')} => {result['reaction_type']}"
                )
            else:
                progress["failed_records"][key] = "model returned unknown"
                save_json_file(REACTION_TYPE_PROGRESS_PATH, progress)
                print(
                    f"[{idx}/{len(pending_keys)}] 未写入结果: {item.get('head')} -> {item.get('tail')} => {UNKNOWN_REACTION_TYPE}"
                )
        except Exception as exc:
            progress["failed_records"][key] = str(exc)
            save_json_file(REACTION_TYPE_PROGRESS_PATH, progress)
            print(f"[{idx}/{len(pending_keys)}] 错误: {item.get('head')} -> {item.get('tail')} | {exc}")
        time.sleep(REACTION_TYPE_SLEEP_SECONDS)

    final_results = load_json_file(REACTION_TYPE_OUTPUT_PATH, [])
    print(f"\n反应类型抽取结果已保存到: {REACTION_TYPE_OUTPUT_PATH}")
    print(f"结果条数: {len(final_results)}")
    print(f"仍待处理条数: {len(progress['pending_records'])}")
    if progress["failed_records"]:
        print(f"失败条数: {len(progress['failed_records'])}")
        print("示例失败:", next(iter(progress["failed_records"].items())))


if __name__ == "__main__":
    run_reaction_type_extraction()

