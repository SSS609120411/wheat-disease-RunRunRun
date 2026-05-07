import streamlit as st
import pandas as pd
import numpy as np
import re
import pywt
import joblib
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
st.set_page_config(page_title="小麦赤霉病反演系统", layout="wide")

#解决字体问题
st.markdown(
    """
    <style>
    /* 全局强制字体 */
    html, body, [class*="css"] {
        font-family: 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif;
    }
    /* 针对 st.dataframe 等表格内部文字 */
    .stDataFrame, .stTable {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    /* 确保侧边栏文字正常 */
    .sidebar .sidebar-content {
        font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 加载模型 ====================
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, 'model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))

# ==================== 植被指数定义 ====================
INDICES = [
    {
        "name": "自定义指数",
        "wavelengths": {"x1": 430, "x2": 676, "x3": 771},
        "calc_func": lambda reflect: np.log(1 + reflect['x3'] / (reflect['x1'] + reflect['x2'] + 1e-8))
    },
    {
        "name": "NDVI",
        "wavelengths": {"red": 660, "nir": 800},
        "calc_func": lambda reflect: (reflect['nir'] - reflect['red']) / (reflect['nir'] + reflect['red'])
    },
    {
        "name": "RVI",
        "wavelengths": {"red": 670, "nir": 780},
        "calc_func": lambda reflect: reflect['nir'] / reflect['red']
    },
    {
        "name": "WFBI",
        "wavelengths": {"r774": 774, "r680": 680, "r521": 521},
        "calc_func": lambda reflect: reflect['r774'] / np.sqrt(reflect['r680']**2 + reflect['r521']**2)
    },
    {
        "name": "RARSc",
        "wavelengths": {"r760": 760, "r500": 500},
        "calc_func": lambda reflect: reflect['r760'] / reflect['r500']
    },
    {
        "name": "SIPI",
        "wavelengths": {"r800": 800, "r445": 445, "r680": 680},
        "calc_func": lambda reflect: (reflect['r800'] - reflect['r445']) / (reflect['r800'] + reflect['r680'])
    },
    {
        "name": "PSSRa",
        "wavelengths": {"r800": 800, "r675": 675},
        "calc_func": lambda reflect: reflect['r800'] / reflect['r675']
    },
    {
        "name": "PSSRb",
        "wavelengths": {"r800": 800, "r650": 650},
        "calc_func": lambda reflect: reflect['r800'] / reflect['r650']
    },
    {
        "name": "GM1",
        "wavelengths": {"r750": 750, "r550": 550},
        "calc_func": lambda reflect: reflect['r750'] / reflect['r550']
    },
    {
        "name": "GM2",
        "wavelengths": {"r750": 750, "r700": 700},
        "calc_func": lambda reflect: reflect['r750'] / reflect['r700']
    },
]

def extract_features_from_dataframe(df):
    band_cols = []
    band_waves = []
    for col in df.columns:
        if col == '样品标号':
            continue
        nums = re.findall(r'\d+\.?\d+', str(col))
        if nums:
            band_cols.append(col)
            band_waves.append(float(nums[0]))

    if len(band_cols) == 0:
        raise ValueError("未找到波段列，请确认Excel列名包含波长数字")

    sorted_idx = np.argsort(band_waves)
    band_cols = [band_cols[i] for i in sorted_idx]
    band_waves = np.array([band_waves[i] for i in sorted_idx])
    spectra_matrix = df[band_cols].values

    all_index_features = []
    for idx_def in INDICES:
        idx_form = {}
        for key, wave in idx_def['wavelengths'].items():
            diffs = np.abs(band_waves - wave)
            best_idx = np.argmin(diffs)
            idx_form[key] = best_idx
        reflect_dict = {}
        for key, idx in idx_form.items():
            reflect_dict[key] = spectra_matrix[:, idx]
        feat = idx_def['calc_func'](reflect_dict)
        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        all_index_features.append(feat)
    feature_indices = np.hstack(all_index_features)

    cwt_targets = [
        {'name': 'WF01', 'scale': 9, 'wave': 393},
        {'name': 'WF02', 'scale': 9, 'wave': 576},
        {'name': 'WF03', 'scale': 10, 'wave': 596},
        {'name': 'WF04', 'scale': 9, 'wave': 624},
        {'name': 'WF05', 'scale': 9, 'wave': 716},
        {'name': 'WF06', 'scale': 7, 'wave': 753},
        {'name': 'WF07', 'scale': 9, 'wave': 959}
    ]
    unique_scales = sorted({t['scale'] for t in cwt_targets})
    coeffs, _ = pywt.cwt(spectra_matrix, scales=unique_scales, wavelet='mexh', axis=-1)
    cwt_features_list = []
    for target in cwt_targets:
        diffs = np.abs(band_waves - target['wave'])
        best_idx = np.argmin(diffs)
        scale_idx = unique_scales.index(target['scale'])
        coeff_series = coeffs[scale_idx, :, best_idx]
        cwt_features_list.append(coeff_series)
    feature_cwt = np.column_stack(cwt_features_list)

    selected_indices = feature_indices[:, [0, 4, 6]]
    selected_cwt = feature_cwt[:, [1, 3, 4, 6]]
    X = np.hstack([selected_indices, selected_cwt])
    return X

# ==================== 病害等级 ====================
def severity_to_level(severity):
    if severity <= 0:
        return "DS=0(无病)"
    elif severity <= 10:
        return "DS=1(轻微)"
    elif severity <= 20:
        return "DS=2(轻度)"
    elif severity <= 30:
        return "DS=3(中度)"
    elif severity <= 40:
        return "DS=4(偏重)"
    else:
        return "DS=5(重度)"

def generate_advice(severity):
    if severity == 0:
        return "健康，无需处理"
    elif severity <= 10:
        return "无大碍，建议继续监测"
    elif severity <= 20:
        return "建议喷施低毒药剂"
    elif severity <= 30:
        return "建议喷施常规药剂"
    elif severity <= 40:
        return "建议立即防治，喷施高效药剂"
    else:
        return "建议立即防治，喷施高效药剂并加强管理"

# ==================== 网页主程序 ====================
def main():
    st.title("🌾 高光谱小麦赤霉病病害严重度反演软件")
    st.markdown("---")

    # 侧边栏说明
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

    # 上传文件
    uploaded_file = st.file_uploader("📂 上传数据文件（.xlsx / .csv）", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("🔍 数据预览（前10行）")
            st.dataframe(df.head(10), use_container_width=True)

            # 开始预测
            if st.button("🚀 开始预测", type="primary"):
                with st.spinner("正在预测中，请稍候..."):
                    X = extract_features_from_dataframe(df)
                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)
                    preds = np.clip(preds, 0, 100)

                # 生成结果
                samples = df["样品标号"].astype(str) if "样品标号" in df.columns else [f"样本{i+1}" for i in range(len(preds))]
                results = []
                for i, sev in enumerate(preds):
                    level = severity_to_level(sev)
                    advice = generate_advice(sev)
                    results.append([samples.iloc[i], round(sev,2), level, advice])

                res_df = pd.DataFrame(results, columns=["样本标识", "严重度(%)", "病害等级", "防治建议"])
                st.subheader("✅ 预测结果")
                st.dataframe(res_df, use_container_width=True)

                # 导出
                csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8-sig')
                st.download_button("💾 导出结果为CSV", csv, "预测结果.csv", "text/csv")

                # 统计图表
                st.markdown("---")
                st.subheader("📊 病害等级统计")
                count = res_df["病害等级"].value_counts().sort_index()
                '''
                fig, ax = plt.subplots(figsize=(10,5))
                count.plot(kind='bar', color=['green','gold','orange','red','purple','blue'], ax=ax)
                plt.xticks(rotation=0)
                st.pyplot(fig)
                '''
                #解决字体问题
                fig, ax = plt.subplots(figsize=(10,5))
                count.plot(kind='bar', color=['green','gold','orange','red','purple','blue'], ax=ax)
                plt.xticks(rotation=0)
                plt.xlabel("病害等级", fontsize=12)
                plt.ylabel("样本数量", fontsize=12)
                plt.title("病害等级统计", fontsize=14)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"出错：{str(e)}")

if __name__ == "__main__":
    main()
