import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from backend import predict_data  # 调用后端

warnings.filterwarnings('ignore')
st.set_page_config(page_title="小麦赤霉病反演系统", layout="wide")

# 解决字体问题
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    .stDataFrame, .stTable {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 界面主程序 ====================
def main():
    st.title("🌾 高光谱小麦赤霉病病害严重度反演软件")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.subheader("📖 使用说明")
        st.info("""
        1. 上传包含高光谱数据的 Excel / CSV
        2. 第一列必须为：样品标号
        3. 波段列名必须包含波长数字
        4. 点击开始预测即可得到结果
        """)
        st.subheader("📊 病害等级标准")
        st.success("""
        DS0：健康
        DS1：轻微
        DS2：轻度
        DS3：中度
        DS4：偏重
        DS5：重度
        """)

    # 上传
    uploaded_file = st.file_uploader("📂 上传数据文件（.xlsx / .csv）", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("🔍 数据预览（前10行）")
            st.dataframe(df.head(10), use_container_width=True)

            # 预测按钮
            if st.button("🚀 开始预测", type="primary"):
                with st.spinner("正在预测中，请稍候..."):
                    res_df = predict_data(df)  # 交给后端计算

                # 展示结果
                st.subheader("✅ 预测结果")
                st.dataframe(res_df, use_container_width=True)

                # 导出
                csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                st.download_button("💾 导出结果为CSV", csv, "预测结果.csv", "text/csv")

                # 图表
                st.markdown("---")
                st.subheader("📊 病害等级统计")
                count = res_df["病害等级"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10,5))
                count.plot(kind='bar', color=['green','gold','orange','red','purple','blue'], ax=ax)
                plt.xticks(rotation=0)
                plt.xlabel("病害等级", fontsize=12)
                plt.ylabel("样本数量", fontsize=12)
                plt.title("病害等级统计", fontsize=14)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"出错：{str(e)}")
    else:
        st.info("请上传 Excel / CSV 文件")

if __name__ == "__main__":
    main()
