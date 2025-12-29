import pickle
import numpy as np

# 생성된 파일명 입력
filename = "crash_worker_3_game_4_20251229_091818.pkl" 

with open(filename, "rb") as f:
    data = pickle.load(f)

print("Error Message:", data['error_message'])
print("Broken Policy (pi):", data['pi']) # 여기에 nan이 잔뜩 섞여 있을 겁니다.
print("Move History Length:", len(data['history']))
# print(data['history']) # 필요하면 전체 수순 출력