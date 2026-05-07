import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from backend import predict_data, get_true_labels, calculate_metrics  # 调用后端
#新加的库
import io

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
    res_df, y_pred = None, None
    # 侧边栏
    #初始化那个东西
    if "y_pred" not in st.session_state:
        st.session_state.y_pred = None
    if "res_df" not in st.session_state:
        st.session_state.res_df = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None   # 用于检测文件是否变化
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
        
        #新加的代码
        st.markdown("---")
        st.subheader("📈 模型精度验证")
        val_file = st.file_uploader("上传带【样品标签】的验证文件", type=["xlsx", "csv"])
        eval_btn = st.button("✅ 开始模型评估")

    # 上传
    uploaded_file = st.file_uploader("📂 上传数据文件（.xlsx / .csv）", type=['xlsx', 'xls', 'csv'])
    #res_df, y_pred = None, None
    
    if uploaded_file is not None:
        #新加了一个清空旧的预测结果
        current_file_name = uploaded_file.name
        if st.session_state.uploaded_file_name != current_file_name:
            st.session_state.y_pred = None
            st.session_state.res_df = None
            st.session_state.uploaded_file_name = current_file_name
        
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
                    res_df, y_pred = predict_data(df)  # 交给后端计算
                    st.session_state.res_df = res_df
                    st.session_state.y_pred = y_pred

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
        # 如果清空了上传文件，也要清空预测结果
        st.session_state.y_pred = None
        st.session_state.res_df = None
        st.session_state.uploaded_file_name = None
    
    #新加的预测功能
    if val_file is not None and eval_btn:
        if st.session_state.y_pred is None:
            st.sidebar.error("请先在上方上传数据文件并点击「开始预测」，生成预测结果后再进行评估！")
        else:
            try:
                # 读取验证文件（真实标签）
                if val_file.name.endswith('.csv'):
                    df_val = pd.read_csv(val_file)
                else:
                    df_val = pd.read_excel(val_file)

                # 获取真实标签
                y_true = get_true_labels(df_val)
                y_pred = st.session_state.y_pred   # 从 session 中获取
            
                # 确保长度一致
                if len(y_true) != len(y_pred):
                    st.error("真实标签数量与预测数量不匹配！")
                else:
                    r2, rmse = calculate_metrics(y_true, y_pred)
                    st.subheader(f"📊 模型评估结果   R² = {r2}   RMSE = {rmse}")

                    # 画图
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    ax.scatter(y_true, y_pred, s=15, color="#2E86AB")
                    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                
                    # 左上角标注
                    ax.text(0.05, 0.95, f"$R^2 = {r2}$\n$RMSE = {rmse}$", 
                            transform=ax.transAxes, fontsize=12, 
                            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white"))
                
                    ax.set_xlabel("真实值")
                    ax.set_ylabel("预测值")
                    st.pyplot(fig)

                    # 下载图片
                    buf = io.BytesIO()
                    fig.savefig(buf, dpi=300, bbox_inches='tight')
                    st.download_button("💾 下载散点图", buf, "scatter.png", "image/png")
                
            except Exception as e:
                st.error(f"评估失败：{str(e)}")
    
    
    
    
if __name__ == "__main__":
    main()
