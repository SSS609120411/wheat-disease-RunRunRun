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

#plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
#plt.rcParams['axes.unicode_minus'] = False

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
        uploaded_file = st.file_uploader("📂 上传数据文件（.xlsx / .csv）", type=['xlsx', 'xls', 'csv'])
    
        # 初始化 session_state 变量
        if "res_df" not in st.session_state:
            st.session_state.res_df = None
        if "y_pred" not in st.session_state:
            st.session_state.y_pred = None
        if "chart_is_bar" not in st.session_state:
            st.session_state.chart_is_bar = True   # True=柱状图, False=饼图
        
        if uploaded_file is not None:
            try:
                # 读取文件
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
                # 上传新文件时清空旧预测结果
                #st.session_state.res_df = None
                #st.session_state.y_pred = None
                
                st.subheader("🔍 数据预览（前10行）")
                st.dataframe(df.head(10), use_container_width=True)
            
                # 预测按钮
                if st.button("🚀 开始预测", type="primary"):
                    with st.spinner("正在预测中，请稍候..."):
                        res_df, y_pred = predict_data(df)
                        st.session_state.res_df = res_df
                        st.session_state.y_pred = y_pred
                
                    st.subheader("✅ 预测结果")
                    st.dataframe(st.session_state.res_df, use_container_width=True)
                    
                    # 导出按钮
                    csv = st.session_state.res_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                    st.download_button("💾 导出CSV", csv, "预测结果.csv", "text/csv")
                    excel_buf = io.BytesIO()
                    st.session_state.res_df.to_excel(excel_buf, index=False, engine="openpyxl")
                    st.download_button("💾 导出Excel", excel_buf, "预测结果.xlsx")
                
                # ========== 图表展示区域（独立于预测按钮） ==========
                if st.session_state.res_df is not None:
                    st.markdown("---")
                    st.subheader("📊 病害等级统计")
                    
                    # 切换图表类型的按钮
                    if st.button("🔄 切换图表类型", key="toggle_chart"):
                        st.session_state.chart_is_bar = not st.session_state.chart_is_bar
                        st.rerun()
                    
                    count = st.session_state.res_df["病害等级"].value_counts().sort_index()
                    
                    if st.session_state.chart_is_bar:
                        # 柱状图
                        levels_order = ["DS=0(无病)", "DS=1(轻微)", "DS=2(轻度)", "DS=3(中度)", "DS=4(偏重)", "DS=5(重度)"]
                        colors = ["#52c41a", "#faad14", "#ff7a45", "#f5222d", "#722ed1", "#1890ff"]
                        counts = []
                        for level in levels_order:
                            cnt = (st.session_state.res_df["病害等级"] == level).sum()
                            counts.append(cnt)
                        fig, ax = plt.subplots(figsize=(10,5))
                        bars = ax.bar(levels_order, counts, color=colors, edgecolor='black')
                        for bar, cnt in zip(bars, counts):
                            height = bar.get_height()
                            # 若高度为0，将文本放在柱子底部上方一点（避免与x轴重叠）
                            y_pos = height + 0.5 if height > 0 else 0.5
                            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                                f'{cnt}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                        ax.set_xlabel("Disease Security", fontsize=12)
                        ax.set_ylabel("Sample Count", fontsize=12)
                        ax.set_title("病害等级统计（柱状图）", fontsize=14)
                        #count.plot(kind='bar', color=['green','gold','orange','red','purple','blue'], ax=ax)
                        #plt.xticks(rotation=0)
                        #plt.xlabel("Disease Security", fontsize=12)
                        #plt.ylabel("Sample Count", fontsize=12)
                        #plt.title("病害等级统计（柱状图）", fontsize=14)
                        st.pyplot(fig)
                    else:
                        # 饼图
                        fig, ax = plt.subplots(figsize=(4,4))
                        colors = ['green','gold','orange','red','purple','blue']
                        count.plot(kind='pie', autopct='%1.1f%%', colors=colors[:len(count)], ax=ax)
                        ax.set_ylabel('')
                        ax.set_title("病害等级统计（饼图）", fontsize=14)
                        st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"出错：{str(e)}")
        else:
            st.info("请上传 Excel / CSV 文件")

    elif page == "使用说明":
        st.subheader("📖 使用说明")
        st.info("""
        1.在点击【开始预测】前可点击其他按钮对软件进行了解
        2.点击【开始预测】进入预测界面
        3.点击【选择Excel/CSV文件】导入数据，可在下方表格中预览数据（仅预览前十列与前二十行）
        4.点击【开始预测】进行病情指数反演，稍后进入结果界面
        5.可在结果界面查看表格中的预测结果
        6.可点击相应按钮实现清空结果、导出结果、返回主页等功能
        """)

    elif page == "病害等级标准":
        st.subheader("📊 病害等级标准")
        
        # 表格数据
        table_data = {
            "病害严重度": ["DS=0", "DS=1", "DS=2", "DS=3", "DS=4", "DS=5"],
            "病情指数": ["DI=0", "0＜DI≤10%", "10%＜DI≤20%", "20%＜DI≤30%", "30%＜DI≤40%", "DI＞40%"],
            "防治建议": [
                "健康，无需处理",
                "无大碍，建议继续监测",
                "建议喷施低毒药剂",
                "建议喷施常规药剂",
                "建议立即防治，喷施高效药剂",
                "建议立即防治，喷施高效药剂并加强管理"
            ]
        }   
        df_std = pd.DataFrame(table_data)
    
        # 使用 st.dataframe 展示表格，可调整宽度
        st.dataframe(df_std, use_container_width=True, hide_index=True)
    
        # 可选：添加额外说明
        st.caption("注：DS=病害严重度等级，DI=病情指数（%）")
        
       
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
