import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from scipy import interpolate, signal
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --- 异常值检测和剔除 ---

def std_dev_outlier(data, threshold=3):
    """
    标准差方法检测异常值
    :param data: 输入数据，一维数组或Series
    :param threshold: 异常值阈值，默认3个标准差
    :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def iqr_outlier(data, k=1.5):
    """
    箱线图 (IQR) 方法检测异常值
    :param data: 输入数据，一维数组或Series
    :param k: IQR倍数，用于计算上下限，默认1.5
    :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def z_score_outlier(data, threshold=3):
    """
    Z分数方法检测异常值
    :param data: 输入数据，一维数组或Series
    :param threshold: Z分数阈值，默认3
    :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    filtered_data = data[np.abs(z_scores) <= threshold]
    return filtered_data


def dbscan_outlier(data, eps=0.5, min_samples=5):
    """
    DBSCAN 算法检测异常值，DBSCAN 将数据点分为核心点、边界点和噪声点，噪声点即为异常值。
    :param data: 输入数据，二维数组或DataFrame
    :param eps: DBSCAN 半径参数，用于定义邻域大小
    :param min_samples: 邻域内的最小样本数，用于定义核心点
    :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    filtered_data = data[labels != -1]
    return filtered_data


def kmeans_outlier(data, n_clusters=3, threshold=2):
    """
        K-means 算法检测异常值，利用聚类中心与数据点的距离判断异常值
        :param data: 输入数据，二维数组或DataFrame
        :param n_clusters: 聚类中心数量
        :param threshold: 异常值距离中心点的阈值
        :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(data)
    distances = kmeans.transform(data)
    labels = np.argmin(distances, axis=1)

    # 计算每个点到对应聚类中心的距离
    distances_to_center = distances[np.arange(len(data)), labels]

    # 选取距离中心点超过阈值的点作为异常值
    filtered_data = data[distances_to_center <= threshold]
    return filtered_data


def isolation_forest_outlier(data, contamination='auto'):
    """
       Isolation Forest 算法检测异常值
        :param data: 输入数据，二维数组或DataFrame
        :param contamination: 异常值比例，默认 'auto'
        :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    isolation_forest = IsolationForest(contamination=contamination, random_state=0)
    labels = isolation_forest.fit_predict(data)
    filtered_data = data[labels == 1]
    return filtered_data


def one_class_svm_outlier(data, nu=0.1, kernel="rbf", gamma="scale"):
    """
       One-Class SVM 算法检测异常值
        :param data: 输入数据，二维数组或DataFrame
        :param nu: 异常值比例，默认0.1
        :param kernel: 核函数类型
        :param gamma: 核函数参数
        :return: 剔除异常值后的数据
    """
    data = np.asarray(data)
    oneclass_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    labels = oneclass_svm.fit_predict(data)
    filtered_data = data[labels == 1]
    return filtered_data


# --- 插值方法 ---
def linear_interpolation(x, y, x_new):
    """
        线性插值，使用直线连接数据点。
        :param x: 原始数据点的 x 坐标
        :param y: 原始数据点的 y 坐标
        :param x_new: 需要插值的新的 x 坐标
        :return: 插值后的 y 值
    """
    f = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    return f(x_new)


def polynomial_interpolation(x, y, x_new, degree=3):
    """
        多项式插值，用一个多项式函数来拟合数据点。
        :param x: 原始数据点的 x 坐标
        :param y: 原始数据点的 y 坐标
        :param x_new: 需要插值的新的 x 坐标
        :param degree: 多项式次数，默认为3
        :return: 插值后的 y 值
    """
    f = interpolate.interp1d(x, y, kind=degree, fill_value="extrapolate")
    return f(x_new)


def spline_interpolation(x, y, x_new, k=3):
    """
       样条插值，使用分段多项式函数来拟合数据点。
       :param x: 原始数据点的 x 坐标
       :param y: 原始数据点的 y 坐标
       :param x_new: 需要插值的新的 x 坐标
       :param k: 样条插值的阶数，默认为3
       :return: 插值后的 y 值
   """


    f = interpolate.splrep(x, y, s=0, k=k)
    return interpolate.splev(x_new, f)


def nearest_neighbor_interpolation(x, y, x_new):
    """
       最近邻插值，使用离插值点最近的数据点的值作为插值结果。
       :param x: 原始数据点的 x 坐标
       :param y: 原始数据点的 y 坐标
       :param x_new: 需要插值的新的 x 坐标
       :return: 插值后的 y 值
   """


    f = interpolate.interp1d(x, y, kind='nearest', fill_value="extrapolate")
    return f(x_new)


def rbf_interpolation(x, y, x_new, function='multiquadric', epsilon=2):
    """
      径向基函数（RBF）插值，使用径向基函数拟合数据点。
      :param x: 原始数据点的 x 坐标
      :param y: 原始数据点的 y 坐标
      :param x_new: 需要插值的新的 x 坐标
      :param function: RBF 函数类型，默认为 'multiquadric'
      :param epsilon: RBF 函数的形状参数
      :return: 插值后的 y 值
  """


    f = interpolate.Rbf(x, y, function=function, epsilon=epsilon)
    return f(x_new)


def fourier_interpolation(x, y, x_new):
    """
      傅里叶插值，通过傅里叶变换在频域中对数据进行插值。
      :param x: 原始数据点的 x 坐标
      :param y: 原始数据点的 y 坐标
      :param x_new: 需要插值的新的 x 坐标
      :return: 插值后的 y 值
  """


    n = len(x)
    y_fft = np.fft.fft(y)
    x_new_n = len(x_new)
    y_new_fft = np.zeros(x_new_n, dtype=complex)
    y_new_fft[:n // 2] = y_fft[:n // 2]
    y_new_fft[-n // 2:] = y_fft[-n // 2:]
    y_new = np.fft.ifft(y_new_fft).real

    # 插值后长度可能与x_new不同，需要截取
    if len(y_new) > len(x_new):
        y_new = y_new[:len(x_new)]

    return y_new


# --- 数据平滑方法 ---
def median_filter(data, window_size=3):
    """
       中值滤波，使用窗口中值替换中心值，可以有效去除脉冲噪声。
       :param data: 输入数据，一维数组或Series
       :param window_size: 窗口大小，必须为奇数，默认为3
       :return: 平滑后的数据
    """
    return signal.medfilt(data, kernel_size=window_size)


def mean_filter(data, window_size=3):
    """
        均值滤波，使用窗口均值替换中心值，可以有效减少随机噪声。
        :param data: 输入数据，一维数组或Series
        :param window_size: 窗口大小，默认为3
        :return: 平滑后的数据
    """
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')


def low_pass_filter(data, cutoff_freq=0.1, order=5):
    """
         低通滤波，滤除高频信号，保留低频信号。
        :param data: 输入数据，一维数组或Series
        :param cutoff_freq: 截止频率，归一化频率
        :param order: 滤波器阶数，默认为5
        :return: 平滑后的数据
    """
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def kalman_filter(data, initial_state=0, initial_covariance=1, process_noise=0.1, measurement_noise=1):
    """
        卡尔曼滤波，通过预测和更新步骤递归估计信号状态。
        :param data: 输入数据，一维数组或Series
        :param initial_state: 初始状态估计值
        :param initial_covariance: 初始状态估计协方差
        :param process_noise: 过程噪声协方差
        :param measurement_noise: 测量噪声协方差
        :return: 平滑后的数据
    """
    n = len(data)
    filtered_state_estimates = np.zeros(n)

    state = initial_state  # 初始状态
    covariance = initial_covariance  # 初始协方差

    for i, measurement in enumerate(data):
        # 预测步骤
        predicted_state = state
        predicted_covariance = covariance + process_noise

        # 更新步骤
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise)
        state = predicted_state + kalman_gain * (measurement - predicted_state)
        covariance = (1 - kalman_gain) * predicted_covariance

        filtered_state_estimates[i] = state

    return filtered_state_estimates


def autoregressive_smoothing(data, p=3, alpha=0.9):
    """
      自回归平滑，通过前几个数据点的值预测当前值。
      :param data: 输入数据，一维数组或Series
      :param p: 自回归模型的阶数，默认为3
      :param alpha: 平滑系数，默认为0.9
      :return: 平滑后的数据
    """
    n = len(data)
    smoothed_data = np.copy(data)
    for t in range(p, n):
        ar_part = np.sum(smoothed_data[t - p:t] * alpha)
        smoothed_data[t] = (1 - alpha) * data[t] + ar_part
    return smoothed_data

# 配置页面
st.set_page_config(layout="wide", page_title="数据预处理分析平台")
st.title("数据预处理/异常值处理")

# ================= 会话状态初始化 =================
if 'outlier_data' not in st.session_state:
    st.session_state.outlier_data = None
if 'outlier_processed' not in st.session_state:
    st.session_state.outlier_processed = None

# ================= 数据配置模块 =================
with st.expander("📊 数据配置", expanded=True):
    data_col1, data_col2 = st.columns([1, 2])

    with data_col1:
        st.subheader("示例数据生成")
        method_type = st.selectbox("选择处理方法类型", ["异常值检测", "插值处理", "数据平滑"])

        # 通用参数配置
        n_samples = st.number_input("样本数量", 50, 1000, 200)
        n_features = st.number_input("特征数量", 1, 10, 3)
        add_noise = st.checkbox("添加噪声")

        # 根据方法类型生成不同示例数据
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        time_axis = np.linspace(0, 10, n_samples)  # 时间轴用于插值和平滑

        if method_type == "异常值检测":
            outlier_idx = np.random.choice(n_samples, size=5, replace=False)
            X[outlier_idx] += 10  # 添加异常值

        elif method_type == "插值处理":
            # 生成带时间维度的数据
            X = np.column_stack([time_axis] + [np.sin(time_axis + i) for i in range(n_features - 1)])

        else:  # 数据平滑
            # 生成带噪声的信号数据
            X = np.column_stack([time_axis] + [
                np.sin(time_axis * (i + 1)) + np.random.normal(0, 0.5, n_samples)
                for i in range(n_features - 1)
            ])

        if add_noise:
            X += np.random.normal(0, 0.3, X.shape)

        if st.button("生成示例数据（不同方法的示例数据可能不同）"):
            columns = ["时间"] + [f"信号_{i}" for i in range(n_features - 1)] if method_type != "异常值检测" \
                else [f"特征_{i}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=columns)
            st.session_state.outlier_data = df
            st.success("示例数据生成成功！")

    with data_col2:
        st.subheader("上传自定义数据")
        uploaded_file = st.file_uploader("选择CSV/Excel文件", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.outlier_data = df
                st.success("数据加载成功！")
            except Exception as e:
                st.error(f"数据加载错误: {str(e)}")

# ================= 数据预览 =================
if st.session_state.outlier_data is not None:
    with st.expander("🔍 数据预览", expanded=True):
        df = st.session_state.outlier_data
        cols = st.columns([3, 1])
        cols[0].dataframe(df.style.highlight_null(color='yellow'), height=300)
        cols[1].markdown(f"**数据维度**: {df.shape}")

# ================= 方法选择与参数配置 =================
method_params = {}
selected_features = []
if st.session_state.outlier_data is not None:
    with st.expander("⚙️ 处理配置", expanded=True):
        df = st.session_state.outlier_data

        # 特征选择
        available_features = df.columns.tolist()
        base_feature = "时间" if method_type != "异常值检测" else None

        selected_features = st.multiselect(
            "选择处理特征（多选）",
            available_features,
            default=[f for f in available_features if f != "时间"]
        )

        # 方法选择
        if method_type == "异常值检测":
            method = st.selectbox("选择检测方法", [
                "标准差法", "IQR法", "Z分数法",
                "DBSCAN", "K-means", "孤立森林", "One-Class SVM"
            ])
            st.text("后四种方法建议多特征同时处理")

            if method in ["标准差法", "IQR法", "Z分数法"]:
                method_params["threshold"] = st.slider("异常阈值", 1.0, 5.0, 3.0)
            elif method == "DBSCAN":
                method_params["eps"] = st.slider("邻域半径", 0.1, 2.0, 0.5)
                method_params["min_samples"] = st.number_input("最小样本数", 2, 20, 5)
            elif method == "K-means":
                method_params["n_clusters"] = st.number_input("聚类数量", 2, 10, 3)
                method_params["threshold"] = st.slider("距离阈值", 1.0, 5.0, 2.0)

        elif method_type == "插值处理":
            method = st.selectbox("选择插值方法", [
                "线性插值", "多项式插值", "样条插值",
                "最近邻插值", "RBF插值", "傅里叶插值"
            ])
            if method == "多项式插值":
                method_params["degree"] = st.number_input("多项式次数", 1, 5, 3)
            elif method == "样条插值":
                method_params["k"] = st.number_input("样条阶数", 1, 5, 3)
            elif method == "RBF插值":
                method_params["function"] = st.selectbox("核函数", [
                    "multiquadric", "gaussian", "inverse", "linear"
                ])

        else:  # 数据平滑
            method = st.selectbox("选择平滑方法", [
                "中值滤波", "均值滤波", "低通滤波",
                "卡尔曼滤波", "自回归平滑"
            ])
            if method == "中值滤波":
                method_params["window_size"] = st.slider("窗口大小", 3, 15, 5, step=2)
            elif method == "低通滤波":
                method_params["cutoff_freq"] = st.slider("截止频率", 0.01, 0.5, 0.1)
            elif method == "自回归平滑":
                method_params["p"] = st.number_input("回归阶数", 1, 10, 3)

# ================= 执行处理 =================
if st.button("开始处理") and st.session_state.outlier_data is not None:
    df = st.session_state.outlier_data
    processed_data = df.copy()

    try:
        for feature in selected_features:
            data = df[feature].values

            if method_type == "异常值检测":
                # 多变量方法单独处理
                if method in ["DBSCAN", "K-means", "孤立森林", "One-Class SVM"]:
                    data_2d = df[selected_features].values

                    if method == "DBSCAN":
                        filtered_idx = dbscan_outlier(data_2d, **method_params)
                    elif method == "K-means":
                        filtered_idx = kmeans_outlier(data_2d, **method_params)
                    elif method == "孤立森林":
                        filtered_idx = isolation_forest_outlier(data_2d)
                    elif method == "One-Class SVM":
                        filtered_idx = one_class_svm_outlier(data_2d)

                     # 生成布尔掩码（改进方案）
                    mask = np.array([np.any(np.all(np.isclose(row, filtered_idx), axis=1))
                               for row in data_2d])
                    processed_data = df[mask]

                else:  # 单变量方法逐个处理
                    valid_indices = set(range(len(df)))  # 初始化有效索引集合

                    for feature in selected_features:
                        data = df[feature].values

                        if method == "标准差法":
                            filtered = std_dev_outlier(data, method_params.get("threshold", 3))
                        elif method == "IQR法":
                            filtered = iqr_outlier(data, method_params.get("k", 1.5))
                        elif method == "Z分数法":
                            filtered = z_score_outlier(data, method_params.get("threshold", 3))

                        # 获取当前特征的合法索引
                        feature_valid = np.isin(df[feature].values, filtered)
                        valid_indices.intersection_update(np.where(feature_valid)[0])

                    # 应用所有特征的共同有效索引
                    processed_data = df.iloc[list(valid_indices)]

            elif method_type == "插值处理":
                # 获取时间轴数据
                time_old = df["时间"].values
                time_new = np.linspace(time_old.min(), time_old.max(), n_samples * 2)

                # 更新处理后的时间轴
                processed_data = pd.DataFrame({"时间": time_new})

                for feature in selected_features:
                    y_old = df[feature].values

                    if method == "线性插值":
                        y_new = linear_interpolation(time_old, y_old, time_new)
                    elif method == "多项式插值":
                        y_new = polynomial_interpolation(time_old, y_old, time_new, method_params.get("degree", 3))
                    elif method == "样条插值":
                        y_new = spline_interpolation(time_old, y_old, time_new, method_params.get("k", 3))
                    elif method == "最近邻插值":
                        y_new = nearest_neighbor_interpolation(time_old, y_old, time_new)
                    elif method == "RBF插值":
                        y_new = rbf_interpolation(time_old, y_old, time_new,
                                                  method_params.get("function", "multiquadric"))
                    else:  # 傅里叶插值
                        y_new = fourier_interpolation(time_old, y_old, time_new)

                    processed_data[feature] = y_new[:len(time_new)]

            else:  # 数据平滑
                for feature in selected_features:
                    data = df[feature].values

                    if method == "中值滤波":
                        smoothed = median_filter(data, method_params.get("window_size", 3))
                    elif method == "均值滤波":
                        smoothed = mean_filter(data, method_params.get("window_size", 3))
                    elif method == "低通滤波":
                        smoothed = low_pass_filter(data, method_params.get("cutoff_freq", 0.1))
                    elif method == "卡尔曼滤波":
                        smoothed = kalman_filter(data)
                    else:  # 自回归平滑
                        smoothed = autoregressive_smoothing(data, method_params.get("p", 3))
                    processed_data[feature] = smoothed

        st.session_state.outlier_processed = processed_data
        st.success("处理完成！")

    except Exception as e:
        st.error(f"处理失败: {str(e)}")

# ================= 结果展示 =================
if st.session_state.outlier_processed is not None:
    with st.expander("📈 处理结果", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("数据对比")
            show_original = st.checkbox("显示原始数据")

            for feature in selected_features:
                df_plot = pd.DataFrame({
                    "原始数据": st.session_state.outlier_data[feature],
                    "处理结果": st.session_state.outlier_processed[feature]
                })
                if not show_original:
                    df_plot = df_plot.drop(columns=["原始数据"])
                st.text(feature)
                st.line_chart(df_plot, height=200)

        with col2:
            st.subheader("数据下载")
            st.session_state.outlier_processed
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.outlier_processed.to_excel(writer, index=False)

            st.download_button(
                label="下载处理结果",
                data=excel_buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )