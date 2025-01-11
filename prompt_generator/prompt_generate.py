### 2024/10/18 酩濠chatgpt人臉評圖生成。
### 說明:此程式用於chatgpt4omini人臉特徵描述生成，提供gpt圖片與prompt(我已經設計好了)，讓gpt對圖片進行描述，並且產生不同的特徵種類(class)，像是:hair、face、race、age等，並產生相應的特徵到folder_name的資料夾內，ex:白人女性會存到white_female資料夾內。
### 使用方法:將待測圖片資料夾放到與此程式同一資料夾內，目前我使用ffgq，在ffgh內有asian_male、asian_female、white_female、white_male、blace_male、black_female，data_set更改你的圖片資料夾名稱，race與gender分別對應子目錄的圖片資料夾。
### 用自己的api_key不要偷用我的摳摳。
### 此程式也提供超參數的調整，可以玩看看呦Ctrl + f 搜尋"temperature"。
### BUG:檔gpt說他不能檢查圖片你重新跑幾次就可以解決。
import cfg
import base64
import requests
import os 
import tiktoken
import time

##############################################################
# OpenAI API Key
api_key = "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"
##############################################################

encoding = tiktoken.encoding_for_model("gpt-4o-mini") 
total_token = 0
total_output_tokens = 0

#########set your data input###############


###########################################

#data_name = f"{race}_{gender}"
# folder_path = f"./{data_set}/{gender}/{race}/{num}"

data_name = "agedb_false" # name
folder_path = "cvpr2025_false_pair/agedb_error_sample_all_id" # input path
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

# Path to your image
for root, sub, files in os.walk(folder_path):
  for filename in files:
    #image_path = r"C:\Users\paul3\Downloads\ffhq\white_male\00015.png"
    md_name = os.path.basename(filename).split('.')[0]
    print(f"Get image {filename}.")
    folder_path = root
    sub = os.path.basename(root)
  ###########################################
    prompt = cfg.prompt(filename)     #<<<<這邊可以改prompt
  ###########################################
    # Getting the base64 string
    base64_image = encode_image(os.path.join(folder_path, filename))
    prompt_tokens = calculate_tokens(prompt)
    image_tokens = calculate_tokens(base64_image)
    total_token += prompt_tokens + image_tokens
    print(f"Prompt token count: {prompt_tokens}")
    print(f"Image token count: {image_tokens}")
    print(f"Total tokens for {filename}: {prompt_tokens + image_tokens}")
    if os.path.isfile(os.path.join(folder_path, filename)):
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
        #################################################
        "temperature": 0.2, # diversity 0~2
        "frequency_penalty": 0.5, # control model using same words -2~2 -2 make model get more same words
        "presence_penalty": -0.2, # control model input new topic -2~2 -2 make model get same topic
        "top_p": 0.9, # sampling rate 0~1 low top_p make low creativity
        "max_tokens": 300
        #################################################
      }
      
      # 取得當前資料夾路徑
      current_folder = os.path.dirname(os.path.abspath(__file__))
      
      # 建立資料夾
      new_folder_path = os.path.join(current_folder, f"{data_name}")
      if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Create folder: {data_name}")
      # 紀錄 API 呼叫的開始時間
      api_start_time = time.time()
    
      # 呼叫 OpenAI API
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      
      # 計算 API 呼叫的end時間
      api_end_time = time.time()
      api_execution_time = api_end_time - api_start_time
      print(f"API call CPU execution time: {api_execution_time:.4f} seconds")

      # 解析API response
      result = response.json()['choices'][0]['message']['content']

      # calculate gpt output tokens
      output_tokens = calculate_tokens(result)
      total_output_tokens += output_tokens
      print(f"Output token count: {output_tokens}")

      #print(result)

  ####################################################################
      # 寫回人臉辨識結果
      os.makedirs(data_name, exist_ok=True)
      output_path = f"{data_name}/{sub}"
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      # 写入到 .txt 文件
      with open(f"{data_name}/{sub}/{md_name}.txt", 'w', encoding='utf-8') as file:
        file.writelines(result)
        print(f"寫入資料夾成功: {data_name}/{sub}/{md_name}.txt")

      # 读取并检查写入内容
      with open(f"{data_name}/{sub}/{md_name}.txt", 'r', encoding='utf-8') as file:
        content = file.read()
        print("寫入文件內容檢查:\n", content)


####################################################################

# 紀錄整體程式的結束 CPU 時間
program_end_time = time.time()
total_program_time = program_end_time - program_start_time

print(f"total_program_time : {total_program_time}sec.")
# 計算使用的token數
print(f"Total input tokens used: {total_token}")
print(f"Total output tokens used: {total_output_tokens}")
print(f"Total tokens used: {total_token + total_output_tokens}")

