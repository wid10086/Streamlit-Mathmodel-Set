import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from io import BytesIO

def mse(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    return mean_absolute_error(y_true, y_pred)


def linear_fit(X, y, X_test):
    X = np.asarray(X)
    y = np.asarray(y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.pinv(X_b).dot(y)
    intercept = theta_best[0]
    coefficients = theta_best[1:]
    y_pred = np.dot(X_test, coefficients) + intercept
    return coefficients, intercept, y_pred


def nonlinear_fit(X, y, X_test, basis_func_type="polynomial", degree=2, interaction_only=False):
    X = np.asarray(X)
    y = np.asarray(y)

    def polynomial_basis(X, degree, interaction_only):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        return poly.fit_transform(X)

    def rbf_basis(X, centers, gamma=0.1):
        from sklearn.metrics.pairwise import rbf_kernel
        return rbf_kernel(X, centers, gamma=gamma)

    if basis_func_type == "polynomial":
        basis_func = lambda X: polynomial_basis(X, degree=degree, interaction_only=False)
    elif basis_func_type == "interaction":
        basis_func = lambda X: polynomial_basis(X, degree=degree, interaction_only=True)
    elif basis_func_type == "rbf":
        from sklearn.cluster import KMeans
        n_centers = min(X.shape[0], 50)
        kmeans = KMeans(n_clusters=n_centers)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        basis_func = lambda X: rbf_basis(X, centers, gamma=0.1)
    else:
        raise ValueError(f"Invalid basis function type: {basis_func_type}")

    X_transformed = basis_func(X)
    X_test_transformed = basis_func(X_test)
    coefficients, intercept, y_pred = linear_fit(X_transformed, y, X_test_transformed)
    return coefficients, intercept, y_pred


def elm_fit(X_train, y_train, X_test, hidden_units=10, activation_func='sigmoid'):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        return np.maximum(0, x)

    def tanh(x):
        return np.tanh(x)

    def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    if activation_func == 'sigmoid':
        activation_func = sigmoid
    elif activation_func == 'relu':
        activation_func = relu
    elif activation_func == 'tanh':
        activation_func = tanh
    elif activation_func == 'leaky_relu':
        activation_func = leaky_relu
    else:
        raise ValueError(f"不支持的激活函数: {activation_func}")

    input_weights = np.random.randn(X_train.shape[1], hidden_units)
    biases = np.random.randn(hidden_units)
    H_train = activation_func(X_train.dot(input_weights) + biases)
    H_test = activation_func(X_test.dot(input_weights) + biases)
    output_weights = np.linalg.pinv(H_train).dot(y_train)
    y_pred_train = H_train.dot(output_weights)
    y_pred_test = H_test.dot(output_weights)
    return output_weights, y_pred_train, y_pred_test


def ridge_regression(X, y, X_test, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    return model.coef_, model.intercept_, y_pred


def kernel_ridge_regression(X_train, y_train, X_test, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0):
    krr = KernelRidge(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, alpha=alpha)
    krr.fit(X_train, y_train)
    return krr.predict(X_test)


def generate_example_data(n_samples, n_features_X, n_features_y, add_noise=False, add_outliers=False):
    """生成多输出示例数据"""
    np.random.seed(42)

    # 生成自变量
    X = np.random.rand(n_samples, n_features_X) * 100

    # 生成系数矩阵（每个y特征对应独立系数）
    true_coef_linear = np.random.randn(n_features_X, n_features_y)  # 线性项系数
    true_coef_nonlinear = np.random.randn(n_features_X, n_features_y)  # 非线性项系数

    # 计算基础y值（包含线性和非线性关系）
    y_linear = X @ true_coef_linear
    y_nonlinear = 0.5 * (X ** 2) @ true_coef_nonlinear
    y = y_linear + y_nonlinear

    # 添加噪声（为每个y特征独立添加）
    if add_noise:
        noise = np.random.normal(0, 10, (n_samples, n_features_y))
        y += noise

    # 添加离群值（按样本添加）
    if add_outliers:
        n_outliers = max(1, n_samples // 10)
        outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
        y[outlier_idx] += np.random.rand(n_outliers, n_features_y) * 200

    # 生成预测数据集
    X_pred = np.random.rand(max(3, n_samples // 2), n_features_X) * 120
    y_pred_linear = X_pred @ true_coef_linear
    y_pred_nonlinear = 0.5 * (X_pred ** 2) @ true_coef_nonlinear
    y_pred_true = y_pred_linear + y_pred_nonlinear

    return X, y, X_pred, y_pred_true


# ================= 页面布局 =================
st.set_page_config(layout="wide")
st.title("📈 多元回归/拟合")

# ================= 数据配置 =================
with st.expander("📊 数据配置", expanded=True):
    data_source = st.radio(
        "请选择数据来源",
        options=["生成示例数据", "上传自定义数据"],
        horizontal=True,
        key="regression_data_source_radio"
    )

    if data_source == "生成示例数据":
        st.subheader("✨ 示例数据设置")
        with st.form("regression_example_config"):
            a1, a2, a3 = st.columns(3)
            n_samples = a1.number_input("样本数量", 3, 1000, 50, key="n_samples")
            n_features_X = a2.number_input("自变量特征数", 1, 20, 5, key="n_features_X")
            n_features_y = a3.number_input("因变量特征数", 1, 10, 1, key="n_features_y")
            x1, x2 = st.columns(2)
            add_noise = x1.checkbox("添加噪声", key="add_noise")
            add_outliers = x2.checkbox("添加离群点", key="add_outliers")
            submit_example = st.form_submit_button("生成示例数据")
            if submit_example:
                X_train, y_train, X_pred, y_true = generate_example_data(
                    n_samples, n_features_X, n_features_y, add_noise, add_outliers
                )
                st.session_state.update({
                    "regression_X_train": X_train, "regression_y_train": y_train,
                    "regression_X_pred": X_pred, "regression_y_true": y_true,
                    "regression_use_example": True
                })
                st.success("示例数据生成成功！")

    elif data_source == "上传自定义数据":
        st.subheader("📤 上传自定义数据")
        uploaded_file = []
        sheet_name = []
        x = ["X_train", "y_train", "X_pred", "y_true(可选)"]
        with st.form("regression_upload_form"):
            for i in range(4):
                row = st.columns([3, 1])
                uploaded_file.append(row[0].file_uploader(f"上传 {x[i]} 数据文件（Excel）", type=["xlsx", "xls"], key=f"upload_{i}")
                )
                sheet_name.append(row[1].text_input(f"{x[i]} 工作表名（Excel）", value="Sheet1", key=f"sheet_{i}"))
            submit_upload = st.form_submit_button("文件上传")
            if submit_upload:
                if uploaded_file[0] and uploaded_file[1] and uploaded_file[2]:
                    try:
                        X_train = pd.read_excel(uploaded_file[0], sheet_name=sheet_name[0])
                        y_train = pd.read_excel(uploaded_file[1], sheet_name=sheet_name[1])
                        X_pred = pd.read_excel(uploaded_file[2], sheet_name=sheet_name[2])
                        if uploaded_file[3]:
                            y_true = pd.read_excel(uploaded_file[3], sheet_name=sheet_name[3])
                        else:
                            y_true = None
                        st.session_state.update({
                            "regression_X_train": X_train, "regression_y_train": y_train,
                            "regression_X_pred": X_pred, "regression_y_true": y_true,
                            "regression_use_example": False
                        })
                        st.success("数据加载成功!")
                    except Exception as e:
                        st.error(f"数据加载失败: {str(e)}")
                else:
                    st.warning("请上传 X_train、y_train 和 X_pred 文件（y_true 可选）")

# 数据初始化（如无数据则默认生成示例数据）
if "regression_X_train" not in st.session_state or "regression_y_train" not in st.session_state or "regression_X_pred" not in st.session_state:
    X_train, y_train, X_pred, y_true = generate_example_data(50, 5, 1)
    st.session_state.update({
        "regression_X_train": X_train, "regression_y_train": y_train,
        "regression_X_pred": X_pred, "regression_y_true": y_true,
        "regression_use_example": True
    })

# ================= 数据预览 =================
with st.expander("👀 数据预览"):
    if "regression_X_train" not in st.session_state:
        st.warning("请先生成或上传数据")
        st.stop()
    cols = st.columns(2)
    with cols[0]:
        st.subheader("📝 训练数据")
        st.write(f"自变量维度: {st.session_state.regression_X_train.shape}")
        st.write(f"因变量维度: {st.session_state.regression_y_train.shape}")
        if st.session_state.regression_use_example:
            st.dataframe(pd.DataFrame(
                st.session_state.regression_X_train,
                columns=[f"特征{i + 1}" for i in range(st.session_state.regression_X_train.shape[1])]
            ))
            df_y_train = pd.DataFrame(
                st.session_state.regression_y_train,
                columns=[f"目标值{i + 1}" for i in range(st.session_state.regression_y_train.shape[1])]
            )
            st.dataframe(df_y_train)
        else:
            st.dataframe(st.session_state.regression_X_train)
            df_y_train = st.session_state.regression_y_train
            st.dataframe(st.session_state.regression_y_train)
    with cols[1]:
        st.subheader("🔮 预测数据")
        st.write(f"自变量维度: {st.session_state.regression_X_pred.shape}")
        if st.session_state.regression_y_true is not None: 
            st.write(f"真实值维度: {st.session_state.regression_y_true.shape}")
        if st.session_state.regression_use_example:
            st.dataframe(pd.DataFrame(
                st.session_state.regression_X_pred,
                columns=[f"特征{i + 1}" for i in range(st.session_state.regression_X_pred.shape[1])]
            ))
            st.dataframe(pd.DataFrame(
                st.session_state.regression_y_true,
                columns=[f"目标值{i + 1}" for i in range(st.session_state.regression_y_true.shape[1])]
            ))
        else:
            st.dataframe(st.session_state.regression_X_pred)
            st.dataframe(st.session_state.regression_y_true)

# ================= 方法配置与参数配置 =================
st.markdown("---")
st.subheader("⚙️ 拟合方法与参数配置")
methods = {
    "线性拟合": {"func": linear_fit, "params": {}},
    "岭回归": {"func": ridge_regression, "params": {"alpha": 1.0}},
    "核岭回归": {"func": kernel_ridge_regression, "params": {
        "kernel": "rbf", "gamma": None, "degree": 3, "coef0": 1, "alpha": 1.0}},
    "ELM拟合": {"func": elm_fit, "params": {"hidden_units": 10, "activation_func": "sigmoid"}},
    "非线性拟合": {"func": nonlinear_fit, "params": {
        "basis_func_type": "polynomial", "degree": 2, "interaction_only": False}}
}

selected_methods = st.multiselect(
    "🧩 选择拟合方法（可多选）",
    options=list(methods.keys()),
    default=["线性拟合", "岭回归"]
)


method_params = {}
for method in selected_methods:
    if method == "岭回归":
        with st.expander(f"{method}参数配置"):
            alpha = st.number_input("岭回归 正则化强度 (alpha)", 0.0, 100.0, 1.0, key=f"{method}_alpha")
            method_params[method] = {"alpha": alpha}
    elif method == "核岭回归":
        with st.expander(f"{method}参数配置"):
            cols = st.columns(2)
            kernel = cols[0].selectbox("核岭回归 核函数", ["rbf", "linear", "poly"], key=f"{method}_kernel")
            gamma = cols[1].number_input("核岭回归 Gamma值", 0.0, 10.0, 1.0, key=f"{method}_gamma")
            method_params[method] = {"kernel": kernel, "gamma": gamma, "alpha": 1.0}
    elif method == "ELM拟合":
        with st.expander(f"{method}参数配置"):
            cols = st.columns(2)
            hidden_units = cols[0].number_input("ELM拟合 隐藏层单元数", 1, 500, 10, key=f"{method}_units")
            activation_func = cols[1].selectbox("ELM拟合 激活函数", ["sigmoid", "relu", "tanh"], key=f"{method}_act")
            method_params[method] = {"hidden_units": hidden_units, "activation_func": activation_func}
    elif method == "非线性拟合":
        with st.expander(f"{method}参数配置"):
            cols = st.columns(3)
            basis_func_type = cols[0].selectbox("非线性拟合 基函数类型", ["polynomial", "interaction", "rbf"], key=f"{method}_basis")
            degree = cols[1].number_input("非线性拟合 多项式次数", 1, 10, 2, key=f"{method}_degree")
            interaction_only = cols[2].checkbox("非线性拟合 仅交互项", key=f"{method}_interaction")
            method_params[method] = {"basis_func_type": basis_func_type, "degree": degree, "interaction_only": interaction_only}

# ================= 拟合执行部分 =================
st.markdown("---")
st.subheader("🚀 拟合与评估")
run_fit = st.button("🚀 运行拟合")

judge = {}
if run_fit:
    if "regression_X_train" not in st.session_state:
        st.warning("请先生成或上传数据")
        st.stop()
    results = {}
    for method in selected_methods:
        try:
            params = method_params.get(method, {})
            if method == "线性拟合":
                if st.session_state.regression_use_example:
                    coefficients, intercept, y_pred = linear_fit(
                        st.session_state.regression_X_train, st.session_state.regression_y_train,
                        st.session_state.regression_X_pred
                    )
                else:
                    coefficients, intercept, y_pred = linear_fit(
                        st.session_state.regression_X_train.values, st.session_state.regression_y_train.values,
                        st.session_state.regression_X_pred.values
                    )
                with st.expander("线性拟合", expanded=False):
                    if st.session_state.regression_y_true is not None:
                        judge[method] = [mse(st.session_state.regression_y_true, y_pred),
                                            rmse(st.session_state.regression_y_true, y_pred),
                                            mae(st.session_state.regression_y_true, y_pred)]
                        st.text(f"MSE:{mse(st.session_state.regression_y_true, y_pred)}，RMSE:{rmse(st.session_state.regression_y_true, y_pred)}，MAE:{mae(st.session_state.regression_y_true, y_pred)}")
                    else:
                        st.text("无y_true，无评估指标")
                    st.text(f"系数：{coefficients}，截距：{intercept}")
            elif method == "岭回归":
                if st.session_state.regression_use_example:
                    coefficients, intercept, y_pred = ridge_regression(
                        st.session_state.regression_X_train, st.session_state.regression_y_train,
                        st.session_state.regression_X_pred, **params)
                else:
                    coefficients, intercept, y_pred = ridge_regression(
                        st.session_state.regression_X_train.values, st.session_state.regression_y_train.values,
                        st.session_state.regression_X_pred.values, **params)
                with st.expander("岭回归", expanded=False):
                    if st.session_state.regression_y_true is not None:
                        judge[method] = [mse(st.session_state.regression_y_true, y_pred),
                                            rmse(st.session_state.regression_y_true, y_pred),
                                            mae(st.session_state.regression_y_true, y_pred)]
                        st.text(f"MSE:{mse(st.session_state.regression_y_true, y_pred)}，RMSE:{rmse(st.session_state.regression_y_true, y_pred)}，MAE:{mae(st.session_state.regression_y_true, y_pred)}")
                    else:
                        st.text("无y_true，无评估指标")
                    st.text(f"系数：{coefficients}，截距：{intercept}")
            elif method == "核岭回归":
                if st.session_state.regression_use_example:
                    y_pred = kernel_ridge_regression(
                        st.session_state.regression_X_train, st.session_state.regression_y_train,
                        st.session_state.regression_X_pred, **params)
                else:
                    y_pred = kernel_ridge_regression(
                        st.session_state.regression_X_train.values, st.session_state.regression_y_train.values,
                        st.session_state.regression_X_pred.values, **params)
                with st.expander("核岭回归", expanded=False):
                    if st.session_state.regression_y_true is not None:
                        judge[method] = [mse(st.session_state.regression_y_true, y_pred),
                                            rmse(st.session_state.regression_y_true, y_pred),
                                            mae(st.session_state.regression_y_true, y_pred)]
                        st.text(f"MSE:{mse(st.session_state.regression_y_true, y_pred)}，RMSE:{rmse(st.session_state.regression_y_true, y_pred)}，MAE:{mae(st.session_state.regression_y_true, y_pred)}")
                    else:
                        st.text("无y_true，无评估指标")
            elif method == "ELM拟合":
                if st.session_state.regression_use_example:
                    output_weights, _, y_pred = elm_fit(
                        st.session_state.regression_X_train, st.session_state.regression_y_train,
                        st.session_state.regression_X_pred, **params)
                else:
                    output_weights, _, y_pred = elm_fit(
                        st.session_state.regression_X_train.values, st.session_state.regression_y_train.values,
                        st.session_state.regression_X_pred.values, **params)
                with st.expander("ELM拟合", expanded=False):
                    if st.session_state.regression_y_true is not None:
                        judge[method] = [mse(st.session_state.regression_y_true, y_pred),
                                            rmse(st.session_state.regression_y_true, y_pred),
                                            mae(st.session_state.regression_y_true, y_pred)]
                        st.text(f"MSE:{mse(st.session_state.regression_y_true, y_pred)}，RMSE:{rmse(st.session_state.regression_y_true, y_pred)}，MAE:{mae(st.session_state.regression_y_true, y_pred)}")
                    else:
                        st.text("无y_true，无评估指标")
                    st.text(f"输出权重：{output_weights}")
            elif method == "非线性拟合":
                if st.session_state.regression_use_example:
                    coefficients, intercept, y_pred = nonlinear_fit(
                        st.session_state.regression_X_train, st.session_state.regression_y_train,
                        st.session_state.regression_X_pred, **params)
                else:
                    coefficients, intercept, y_pred = nonlinear_fit(
                        st.session_state.regression_X_train.values, st.session_state.regression_y_train.values,
                        st.session_state.regression_X_pred.values, **params)
                with st.expander("非线性拟合", expanded=False):
                    if st.session_state.regression_y_true is not None:
                        judge[method] = [mse(st.session_state.regression_y_true, y_pred),
                                            rmse(st.session_state.regression_y_true, y_pred),
                                            mae(st.session_state.regression_y_true, y_pred)]
                        st.text(f"MSE:{mse(st.session_state.regression_y_true, y_pred)}，RMSE:{rmse(st.session_state.regression_y_true, y_pred)}，MAE:{mae(st.session_state.regression_y_true, y_pred)}")
                    else:
                        st.text("无y_true，无评估指标")
                    st.text(f"系数：{coefficients}，截距：{intercept}")
            results[method] = y_pred
            st.success(f"{method} 拟合完成!")
        except Exception as e:
            st.error(f"{method} 拟合失败: {str(e)}")
    # 评判指标区块
    with st.expander("📊 评判指标", expanded=False):
        if len(judge):
            df = pd.DataFrame(judge, index=["MSE", "RMSE", "MAE"])
            st.dataframe(df)
            st.line_chart(df, use_container_width=True)
        else:
            st.text("无y_true，无评估指标")
    if results:
        st.session_state.regression_results = results

# ================= 结果导出部分 =================
st.markdown("---")
if 'regression_results' in st.session_state:
    st.subheader("📥 结果导出")
    result_methods = [m for m in st.session_state.regression_results if m != "真实值"]
    cols = st.columns(len(result_methods))
    feature_names = df_y_train.columns.tolist() if 'df_y_train' in locals() else None
    for idx, method in enumerate(result_methods):
        data = st.session_state.regression_results[method]
        with cols[idx]:
            df_result = pd.DataFrame(data, columns=feature_names)
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Result')
            excel_bytes = excel_buffer.getvalue()
            st.download_button(
                label=f"⬇️ 下载 {method} 结果",
                data=excel_bytes,
                file_name=f"{method}_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"btn_{method}"
            )
            if st.checkbox(f"👁️ 显示{method}结果", key=f"cb_{method}"):
                st.dataframe(df_result)