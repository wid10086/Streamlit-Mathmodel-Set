import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO


# 归一化方法定义保持不变，此处省略以节省篇幅
# [原有所有归一化函数定义...]
def min_max_normalize(data, feature_range=(0, 1)):
    """
    最小-最大归一化

    :param data: 需要归一化的输入数据，数组类型
    :param feature_range: 归一化后数据的范围，元组 (min, max)，默认=(0, 1)
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    std = (data - min_val) / (max_val - min_val)
    scaled = std * (feature_range[1] - feature_range[0]) + feature_range[0]

    return scaled

def z_score_normalize(data):
    """
    Z-score标准化

    :param data: 需要归一化的输入数据，数组类型
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return (data - mean) / std

def max_abs_normalize(data):
    """
    最大绝对值归一化

    :param data: 需要归一化的输入数据，数组类型
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    max_abs = np.max(np.abs(data), axis=0)

    return data / max_abs

def decimal_scaling_normalize(data):
    """
    小数定标归一化

    :param data: 需要归一化的输入数据，数组类型
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    max_val = np.max(np.abs(data), axis=0)
    j = np.ceil(np.log10(max_val)).astype(int)

    return data / (10 ** j)

def vector_normalize(data):
    """
    向量归一化(L2范数归一化)

    :param data: 需要归一化的输入数据，数组类型
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    norm = np.linalg.norm(data, axis=1, keepdims=True)

    return data / norm

def log_normalize(data, base=np.e):
    """
    对数归一化

    :param data: 需要归一化的输入数据，数组类型
    :param base: 对数的底数，默认=e
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    if np.any(data <= 0):
        st.text("对数归一化要求所有值必须为正")
        raise ValueError("对数归一化要求所有值必须为正")
    return np.log(data) / np.log(base)

def exp_normalize(data):
    """
    指数归一化

    :param data: 需要归一化的输入数据，数组类型
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    return np.exp(data - np.max(data, axis=0))

def softmax_normalize(data, axis=0):
    """
    Softmax归一化

    :param data: 需要归一化的输入数据，数组类型
    :param axis: 进行归一化的轴，默认=0
    :return: 归一化后的数据，形状与输入数据相同
    """
    data = np.asarray(data)
    exp_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def generate_example_data(n_samples, n_features, add_noise=False, add_outliers=False):
    """生成示例数据，包含可选噪声和离群点"""
    np.random.seed(42)
    base_data = np.random.rand(n_samples, n_features) * 100

    if add_noise:
        noise = np.random.normal(0, 10, (n_samples, n_features))
        base_data += noise

    if add_outliers:
        n_outliers = max(1, n_samples // 10)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        base_data[outlier_indices] += np.random.rand(n_outliers, n_features) * 500

    return base_data


# 页面布局设置
#st.set_page_config(layout="wide")
st.title("数据归一化/标准化")

# ================= 数据输入部分 =================
with st.expander("数据配置", expanded=True):
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("上传自定义数据")
        with st.form("you_config"):
            uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"],
                                         help="每行一个样本，每列一个特征")
            sheet_name = st.text_input("请输入工作表名称或索引，默认值为“Sheet1”", value="Sheet1")
            if st.form_submit_button("上传数据"):
                try:
                    st.text(f"uploaded_file：{uploaded_file.name}，sheet_name：{sheet_name}")
                except Exception as e:
                    st.error(f"报错: {str(e)}")

    with col2:
        st.subheader("生成示例数据")
        with st.form("example_config"):
            n_samples = st.number_input("样本数量", min_value=3, value=50)
            n_features = st.number_input("特征数量", min_value=1, value=5)
            add_noise = st.checkbox("添加噪声")
            add_outliers = st.checkbox("添加离群点")
            if st.form_submit_button("生成数据"):
                example_data = generate_example_data(n_samples, n_features, add_noise, add_outliers)
                st.session_state.normalize_current_data = example_data
                st.session_state.normalize_data_source = "example"

# 加载数据
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file,sheet_name = sheet_name)
        st.session_state.normalize_current_data = df.values
        st.session_state.normalize_data_source = "uploaded"
    except Exception as e:
        st.error(f"文件读取错误: {str(e)}")
elif 'normalize_current_data' not in st.session_state:
    # 初始化默认数据
    st.session_state.normalize_current_data = generate_example_data(50, 5)
    st.session_state.normalize_data_source = "example"

# 显示当前数据
st.subheader("当前数据预览")
with st.expander(f"当前数据", expanded=True):
    if st.session_state.normalize_data_source == "uploaded":
        preview_df = df
        st.dataframe(df)
    else:
        preview_df = pd.DataFrame(
            st.session_state.normalize_current_data,
            columns=[f"特征{i + 1}" for i in range(st.session_state.normalize_current_data.shape[1])]
        )
        st.dataframe(preview_df)

# ================= 归一化配置部分 =================
st.markdown("---")
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("归一化配置")
    methods = {
        "最小-最大归一化": {"func": min_max_normalize, "params": {"feature_range": (0, 1)}},
        "Z-Score标准化": {"func": z_score_normalize},
        "最大绝对值归一化": {"func": max_abs_normalize},
        "小数定标归一化": {"func": decimal_scaling_normalize},
        "向量归一化(L2范数)": {"func": vector_normalize},
        "对数归一化": {"func": log_normalize, "params": {"base": np.e}},
        "Softmax归一化": {"func": softmax_normalize}
    }

    selected_methods = st.multiselect(
        "选择归一化方法",
        options=list(methods.keys()),
        default=["最小-最大归一化", "Z-Score标准化"]
    )

    # 动态参数配置
    method_params = {}
    for method in selected_methods:
        if method == "最小-最大归一化":
            with st.expander(f"{method}参数配置"):
                col_min, col_max = st.columns(2)
                min_val = col_min.number_input("最小值", value=0.0, key=f"{method}_min")
                max_val = col_max.number_input("最大值", value=1.0, key=f"{method}_max")
                method_params[method] = {"feature_range": (min_val, max_val)}

        if method == "对数归一化":
            with st.expander(f"{method}参数配置"):
                base = st.number_input("对数基数", value=np.e, key=f"{method}_base")
                method_params[method] = {"base": base}

# ================= 可视化部分 =================
with col2:
    st.subheader("可视化结果")

    if st.button("运行归一化"):
        data = st.session_state.normalize_current_data
        results = {"原始数据": data}

        for method in selected_methods:
            try:
                params = method_params.get(method, {})
                if method == "Softmax归一化":
                    normalized = methods[method]["func"](data, axis=0)
                else:
                    normalized = methods[method]["func"](data, **params)
                results[method] = normalized
            except Exception as e:
                st.error(f"{method}执行失败: {str(e)}")

        st.session_state.normalize_results = results

    if 'normalize_results' in st.session_state:
        feature_names = preview_df.columns.tolist()
        selected_feature = st.selectbox("选择要可视化的特征", feature_names)
        feature_idx = feature_names.index(selected_feature)

        # 添加原始数据显示开关
        show_raw = st.checkbox("显示原始数据", value=True, key='show_raw_data')

        # 准备数据
        chart_data = {}
        for method, data in st.session_state.normalize_results.items():
            if method == "原始数据" and not show_raw:
                continue
            chart_data[method] = data[:, feature_idx]

        # 转换为DataFrame
        chart_df = pd.DataFrame(chart_data)
        chart_df.index.name = "样本索引"

        # 使用Streamlit内置折线图
        st.line_chart(chart_df, use_container_width=True)

# ================= 结果导出部分 =================
st.markdown("---")
if 'normalize_results' in st.session_state:
    st.subheader("结果导出")
    cols = st.columns(len(st.session_state.normalize_results))

    for idx, (method, data) in enumerate(st.session_state.normalize_results.items()):
        if method == "原始数据":
            continue

        with cols[idx - 1]:
            df_result = pd.DataFrame(data, columns=feature_names)
            # 生成Excel字节流
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Result')
            excel_bytes = excel_buffer.getvalue()
            st.download_button(
                label=f"下载 {method} 结果",
                data=excel_bytes,
                file_name=f"{method}_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"btn_{method}"
            )

            if st.checkbox(f"显示{method}结果", key=f"cb_{method}"):
                st.dataframe(df_result)