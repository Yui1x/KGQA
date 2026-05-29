





import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
import urllib3
from pathlib import Path


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'http://www.ichemistry.cn/',
    'Connection': 'keep-alive',
}

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "数据集预处理").exists() and (PROJECT_ROOT.parent.parent / "数据集预处理").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent.parent

OUTPUT_DIR = PROJECT_ROOT / "数据集预处理" / "MSDS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_msds_structured(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    result = {"标题": "", "reference": url, "sections": {}}
    


    title_element = soup.find(text=re.compile(r'msds目录'))
    if title_element:

        clean_title = re.sub(r'msds目录.*', '', title_element).strip()
        result["标题"] = clean_title


    started = False
    current_section = None
    

    rows = soup.find_all('tr')
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        cell_texts = [c.get_text(strip=True) for c in cells]
        row_text = "".join(cell_texts)


        if "其他化学品msds报告" in row_text:
            break


        if "第一部分" in row_text and "：" in row_text:
            started = True

        if not started:
            continue


        section_match = re.search(r'(第[一二三四五六七八九十\d]+部分[：:]\s*[\u4e00-\u9fa5/]+)', row_text)
        if section_match:
            current_section = section_match.group(1).strip()

            current_section = current_section.replace("回目录", "").strip()
            result["sections"][current_section] = {}
            continue



        if current_section and len(cell_texts) >= 2:

            for i in range(0, len(cell_texts) - 1, 2):
                key = cell_texts[i].rstrip('：:').strip()
                val = cell_texts[i+1].strip()
                

                if key and val and "部分" not in key and "msds" not in key.lower():
                    result["sections"][current_section][key] = val

    return result

def main():
    session = requests.Session()
    success_count = 0
    fail_count = 0
    

    for pid in [423, 991, 1013, 1131, 1183, 1208, 1230, 1457, 1488, 1522, 1525, 1527, 1537, 1605, 1608, 1706, 2125, 2208, 2274, 2439, 2442, 2614, 2700, 2944]:
        url = f"http://www.ichemistry.cn/msds/{pid}.htm"
        print(f"[{pid}/2967] 正在抓取: {url}", end=" ")
        
        try:

            resp = session.get(url, headers=HEADERS, timeout=15, verify=False)
            

            if resp.status_code == 403:
                print("-> 触发 403 反爬，暂停 10 秒并跳过...")
                time.sleep(10)
                continue
            
            if resp.status_code != 200:
                print(f"-> 跳过 (状态码: {resp.status_code})")
                continue


            resp.encoding = resp.apparent_encoding 
            

            data = extract_msds_structured(resp.text, url)
            

            if not data["sections"]:
                print("-> 解析为空，可能是页面不存在")
                continue


            raw_name = data["标题"] if data["标题"] else f"chemical_{pid}"
            file_name = re.sub(r'[\\/:*?"<>|]', '_', raw_name)
            file_path = os.path.join(OUTPUT_DIR, f"{file_name}.json")
            

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            print(f"-> 成功保存: {file_name}.json")
            
        except Exception as e:
            fail_count += 1
            print(f"-> 报错: {e}")
        

        time.sleep(0.5)

    print(f"\n{'='*30}\n抓取任务完成！\n成功: {success_count}\n失败: {fail_count}\n{'='*30}")

if __name__ == "__main__":
    main()





import os
import json

def batch_rename_and_update_json(folder_path):

    punctuation_map = str.maketrans({
        '，': ',',
        '。': '.',
        '（': '(',
        '）': ')',
        '：': ':',
        '“': '"',
        '”': '"'
    })

    modified_files = []


    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):

            old_name_base = os.path.splitext(filename)[0]
            

            new_name_base = old_name_base.translate(punctuation_map)
            

            if old_name_base == new_name_base:
                continue

            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, f"{new_name_base}.json")

            try:

                with open(old_file_path, 'r', encoding='utf-8') as f:
                    content_str = f.read()
                


                new_content_str = content_str.replace(old_name_base, new_name_base)
                

                with open(old_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_str)
                
                os.rename(old_file_path, new_file_path)
                
                modified_files.append({
                    "old_name": filename,
                    "new_name": f"{new_name_base}.json"
                })
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    return modified_files


target_folder = str(PROJECT_ROOT / "数据集预处理" / "MSDS")
result = batch_rename_and_update_json(target_folder)
print("已调整的文件：", result)

