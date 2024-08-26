# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 加载模型
model = joblib.load('california_model.pkl')

# 应用标题
st.title("模拟房价预测")

# 用户输入
st.header("输入房屋特征:")

# 使用number_input来接收用户输入
MedInc = st.number_input('收入中位数', 0.0, 10.0, 5.0)
HouseAge = st.number_input('房龄中位数', 0.0, 50.0, 25.0)
AveRooms = st.number_input('每户平均房间数', 0.0, 10.0, 5.0)
AveBedrms = st.number_input('每户平均卧室数', 0.0, 5.0, 2.5)
Population = st.number_input('人口数量', 0.0, 5000.0, 2500.0)
AveOccup = st.number_input('每户平均居住人数', 0.0, 10.0, 5.0)
Latitude = st.number_input('纬度', 32.0, 42.0, 37.0)
Longitude = st.number_input('经度', -120.0, -110.0, -115.0)

# 添加预测按钮
if st.button('预测房价'):
    # 将输入数据转换为DataFrame
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_df = pd.DataFrame(input_data, columns=['收入中位数', '房龄中位数', '每户平均房间数', '每户平均卧室数', '人口数量', '每户平均居住人数', '纬度', '经度'])

    # 进行预测
    prediction = model.predict(input_df)

    # 显示结果
    st.subheader('预测的房价（单位：美元）:')
    st.write(f"${prediction[0]*100000:.2f}")
