import cfg
import base64
import requests
import os
import tiktoken
import time

##############################################################
# OpenAI API Key
api_key = "sk-proj-xxxxxxx_your_own_api_key_xxxxx"
##############################################################

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
total_token = 0
total_output_tokens = 0

#########set your data input###############
data_name = "agedb_false"  # name
folder_path = "cvpr2025_false_pair/agedb_error_sample_all_id"  # input path
###########################################

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 計算text的token數
def calculate_tokens(text):
    """計算給定文字的 token 數量"""
    tokens = encoding.encode(text)
    return len(tokens)

# 計算執行時間
program_start_time = time.time()

# ---修改區開始---
# 只取各子資料夾中的第一張圖片進行 GPT 呼叫
for root, subfolders, files in os.walk(folder_path):
    # 如果該子資料夾沒有檔案則跳過
    if not files:
        continue
    
    # 只取第一張圖
    filename = files[0]
    md_name = os.path.splitext(filename)[0]  # 去掉副檔名
    print(f"Get image {filename}.")

    # sub 會是該路徑最後一層資料夾名稱
    sub = os.path.basename(root)

    # 這裡可以自由決定要如何產生 prompt，以下先示範保持原先寫法
    prompt = cfg.prompt(filename)  #<<<<這邊可以改prompt
    
    # 取得檔案的 base64
    base64_image = encode_image(os.path.join(root, filename))

    # token 計算
    prompt_tokens = calculate_tokens(prompt)
    image_tokens = calculate_tokens(base64_image)
    total_token += prompt_tokens + image_tokens
    print(f"Prompt token count: {prompt_tokens}")
    print(f"Image token count: {image_tokens}")
    print(f"Total tokens for {filename}: {prompt_tokens + image_tokens}")

    if os.path.isfile(os.path.join(root, filename)):
        print(f"Processing image: {filename}.")
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,          # diversity 0~2
            "frequency_penalty": 0.5,    # -2~2
            "presence_penalty": -0.2,    # -2~2
            "top_p": 0.9,                # 0~1
            "max_tokens": 300
        }

        # 取得當前程式所在資料夾
        current_folder = os.path.dirname(os.path.abspath(__file__))

        # 建立結果輸出資料夾
        new_folder_path = os.path.join(current_folder, f"{data_name}")
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Create folder: {data_name}")

        # 紀錄 API 呼叫的開始時間
        api_start_time = time.time()

        # 呼叫 OpenAI API
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # 計算 API 呼叫結束時間
        api_end_time = time.time()
        api_execution_time = api_end_time - api_start_time
        print(f"API call CPU execution time: {api_execution_time:.4f} seconds")

        # 解析 API 回傳
        result = response.json()['choices'][0]['message']['content']

        # 計算 GPT output tokens
        output_tokens = calculate_tokens(result)
        total_output_tokens += output_tokens
        print(f"Output token count: {output_tokens}")

        # 建立對應的子資料夾 (以 sub 命名)
        output_path = os.path.join(data_name, sub)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 將結果寫入 .txt 檔
        txt_path = os.path.join(output_path, f"{md_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.writelines(result)
            print(f"寫入資料夾成功: {txt_path}")

        # 讀取並檢查
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print("寫入文件內容檢查:\n", content)
# ---修改區結束---

# 紀錄整體程式的結束 CPU 時間
program_end_time = time.time()
total_program_time = program_end_time - program_start_time

print(f"total_program_time : {total_program_time} sec.")
# 計算使用的token數
print(f"Total input tokens used: {total_token}")
print(f"Total output tokens used: {total_output_tokens}")
print(f"Total tokens used: {total_token + total_output_tokens}")
