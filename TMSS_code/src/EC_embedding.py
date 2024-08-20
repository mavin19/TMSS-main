# =========================================================
# -*- coding: utf-8 -*-
# @Time : 2024/6/18 13:00
# @Author : Stefan Wang
# @Email: 787758413@qq.com
# @File : EC_embedding.py
# @Project : CLIP-Driven-Universal-Model-main
# =========================================================
import os
import clip
import torch
import pandas as pd

excel_file = '/home/sribd/Desktop/TMSS_EC_Sorted/EC.csv'
df = pd.read_csv(excel_file)
tl_column = df['TL']
location_column = df['Location']
# t_5category_column = df['TNM_5category']
tumor_features = []
for i in range(len(df)):
    tumor_feature = {
    'TL': tl_column[i],
    'Location': location_column[i],
    # 'T_5category': t_5category_column[i]
    }

    tumor_features.append(tumor_feature)

print(tumor_features)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
descriptions = [f"Tumor with TL: {feature['TL']}, Location: {feature['Location']}" for feature in tumor_features]
# T_5category: {feature['T_5category']
print(descriptions)
text_inputs = torch.cat([clip.tokenize(description) for description in descriptions]).to(device)
print(text_inputs)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding.pth')

