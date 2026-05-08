import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from backend import predict_data, get_true_labels, calculate_metrics  # 调用后端
#新加的库
import io
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
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

    # 侧边栏导航按钮
    with st.sidebar:
        st.subheader("📌 功能导航")
        page = st.radio("", ["预测界面", "使用说明", "病害等级标准", "模型评估"])

    # ==================== 页面内容 ====================
    if page == "预测界面":
        # --------------- 你原来的预测界面 全部放这里 ---------------
        uploaded_file = st.file_uploader("📂 上传数据文件（.xlsx / .csv）", type=['xlsx', 'xls', 'csv'])
        res_df, y_pred = None, None
        
        if uploaded_file is not None:        
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.subheader("🔍 数据预览（前10行）")
                st.dataframe(df.head(10), use_container_width=True)

                if st.button("🚀 开始预测", type="primary"):
                    with st.spinner("正在预测中，请稍候..."):
                        res_df, y_pred = predict_data(df)
                        st.session_state.y_pred = y_pred

                    st.subheader("✅ 预测结果")
                    st.dataframe(res_df, use_container_width=True)

                    csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                    st.download_button("💾 导出CSV", csv, "预测结果.csv", "text/csv")

                    excel_buf = io.BytesIO()
                    res_df.to_excel(excel_buf, index=False, engine="openpyxl")
                    st.download_button("💾 导出Excel", excel_buf, "预测结果.xlsx")

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

    elif page == "使用说明":
        st.subheader("📖 使用说明")
        st.info("""
        1. 上传包含高光谱数据的 Excel / CSV
        2. 第一列必须为：样品标号
        3. 波段列名必须包含波长数字
        4. 点击开始预测即可得到结果
        """)

    elif page == "病害等级标准":
        st.subheader("📊 病害等级标准")
        st.success("""
        DS0：健康
        DS1：轻微
        DS2：轻度
        DS3：中度
        DS4：偏重
        DS5：重度
        """)

    elif page == "模型评估":
        st.subheader("📈 模型精度评估")
        val_file = st.file_uploader("上传带【样品标签】的验证文件", type=["xlsx", "csv"])
        run_eval = st.button("✅ 开始评估", type="primary")

        if val_file is not None and run_eval:
            with st.spinner("📊 评估中，请稍候..."):   # 这里显示提示
                try:
                    if val_file.name.endswith('.csv'):
                        df_val = pd.read_csv(val_file)
                    else:
                        df_val = pd.read_excel(val_file)

                    if "样品标签" not in df_val.columns:
                        st.error("验证文件必须包含 '样品标签' 列")
                    else:
                        #with st.spinner("评估中..."):
                        _, y_pred_val = predict_data(df_val)
                        y_true = df_val["样品标签"].values

                        r2, rmse = calculate_metrics(y_true, y_pred_val)
                        st.subheader(f"📊 评估结果  R²={r2}  RMSE={rmse}")
    
                        fig, ax = plt.subplots(figsize=(8,6), dpi=300)
                        ax.scatter(y_true, y_pred_val, s=15, color="#2E86AB")
                        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                        ax.text(0.05, 0.95, f"$R^2={r2}$\n$RMSE={rmse}$", transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=dict(facecolor="white"))
                        ax.set_xlabel("真实值")
                        ax.set_ylabel("预测值")
                        st.pyplot(fig)

                        buf = io.BytesIO()
                        fig.savefig(buf, dpi=300, bbox_inches='tight')
                        st.download_button("💾 下载散点图", buf, "scatter.png")
                except Exception as e:
                    st.error(f"失败：{str(e)}")
    
    
    
    
if __name__ == "__main__":
    main()
