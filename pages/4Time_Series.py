import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from prophet import Prophet

# 目标代码中的三个预测函数（保持原样）
# ... [这里完整包含目标代码中的 arima_forecast, grey_prediction, prophet_forecast 函数] ...
def arima_forecast(data, p_max=None, d=1, q_max=None, forecast_num=5):
    """
    一步到位的 ARIMA 模型预测函数

    :param data: 时间序列数据，Pandas Series 格式，需要指定索引为时间
    :param p_max: p 的最大值，默认=None，自动设置为序列长度的十分之一
    :param d: 差分阶数，默认=1
    :param q_max: q 的最大值，默认=None，自动设置为序列长度的十分之一
    :param forecast_num: 预测步数，默认=5
    :return: 包含预测值、标准差和置信区间的 DataFrame
    """

    # 数据类型转换
    data = data.astype(float)

    # 差分处理
    if d > 0:
        diff_data = data.diff(periods=d).dropna()
    else:
        diff_data = data

    # ADF 检验
    adf_result = adfuller(diff_data)
    st.text(f'差分序列 ADF 检验结果:')
    st.text(f'  ADF Statistic: {adf_result[0]}')
    st.text(f'  p-value: {adf_result[1]}')
    for key, value in adf_result[4].items():
        st.text(f'  {key}: {value}')

    # 白噪声检验
    lb_result = acorr_ljungbox(diff_data, lags=[1], return_df=True) # lags=1 滞后1阶
    st.text(f'\n差分序列白噪声检验结果:')
    st.text(lb_result)
    # 确保 p-value 大于指定的显著性水平（如 0.05）以拒绝原假设（即序列是白噪声）

    # 确定 p 和 q 的最大值
    if p_max is None:
        p_max = int(len(diff_data) / 10)
    if q_max is None:
        q_max = int(len(diff_data) / 10)

    # 使用 BIC 准则定阶
    bic_matrix = []
    for p in range(p_max + 1):
        tmp = []
        for q in range(q_max + 1):
            try:
                tmp.append(ARIMA(data, order=(p, d, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p, q = bic_matrix.stack().idxmin()
    st.text(f'\nBIC 最小的 p 值和 q 值为: {p}, {q}')

    # 构建 ARIMA 模型
    model = ARIMA(data, order=(p, d, q))

    # 拟合模型
    model_fit = model.fit()

    # 模型总结
    #print(model_fit.summary())

    # 进行预测,使用forecast()函数
    forecast_result = model_fit.get_forecast(steps=forecast_num)
    forecast_value = forecast_result.predicted_mean
    forecast_se = forecast_result.se_mean
    forecast_ci = forecast_result.conf_int()

    # 构建包含预测结果的 DataFrame
    forecast_df = pd.DataFrame({
        'forecast_value': forecast_value,
        'forecast_se': forecast_se,
        'lower_bound': forecast_ci.iloc[:, 0],
        'upper_bound': forecast_ci.iloc[:, 1],
    }, index=pd.date_range(data.index[-1] + pd.DateOffset(1), periods=forecast_num, freq=data.index.freq))
    # 假设数据的频率是固定的，例如每天或每月。如果没有设置频率，可以使用 data.index.freq = 'D'（或其他适当的频率）来设置
    st.text(f"推断出数据的时间频率为: {data.index.freq}")
    return forecast_df

def grey_prediction(data, forecast_steps=1, alpha=0.5):
    """
    完整且全面的灰度预测模型 GM(1,1) 封装函数

    :param data: 时间序列数据，一维数组、列表或 Pandas Series 格式
    :param forecast_steps: 预测步数，默认为 1
    :param alpha: 紧邻均值生成系数，取值范围 (0, 1)，默认为 0.5
    :return: 包含原始数据、拟合值和预测值的 DataFrame
             以及模型参数 a (发展系数) 和 b (灰作用量)
    """

    # 1. 数据预处理
    # 1.1 转换为 NumPy 数组
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("输入数据必须是一维的")

    # 1.2 检查数据是否包含非正值
    if np.any(data <= 0):
        raise ValueError("灰度预测模型要求数据为非负值")

    # 2. 累加生成
    x1 = np.cumsum(data)

    # 3. 紧邻均值生成
    z1 = alpha * x1[1:] + (1 - alpha) * x1[:-1]

    # 4. 构造数据矩阵 B 和数据向量 Y
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = data[1:].reshape((len(data) - 1, 1))

    # 5. 使用最小二乘法计算参数 a 和 b
    try:
        a, b = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    except np.linalg.LinAlgError:
        st.text("警告：矩阵 (B^T * B) 不可逆，可能存在共线性问题。")
        # 使用伪逆来解决
        a, b = np.dot(np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T), Y)

    # 6. 建立预测模型
    def predict(k):
        return (data[0] - b / a) * np.exp(-a * (k)) + b / a

    # 7. 计算拟合值
    fitted_values = np.zeros(len(data))
    fitted_values[0] = data[0]  # 第一个值为原始值
    for i in range(1, len(data)):
        fitted_values[i] = predict(i) - predict(i - 1)

    # 8. 进行预测
    forecast_values = np.zeros(forecast_steps)
    for i in range(forecast_steps):
        forecast_values[i] = predict(len(data) + i) - predict(len(data) + i - 1)

    # 9. 误差检验
    # 9.1 残差
    residuals = data - fitted_values
    # 9.2 相对误差
    relative_errors = np.abs(residuals / data)
    # 9.3 平均相对误差
    mean_relative_error = np.mean(relative_errors)
    # 9.4 级比
    ratio = data[1:] / data[:-1]
    # 9.5 级比偏差
    rho = 1 - (1 - 0.5 * a) / (1 + 0.5 * a)
    ratio_deviation = np.abs(ratio - np.exp(-a))
    # 9.6 平均级比偏差
    mean_ratio_deviation = np.mean(ratio_deviation)

    st.text("模型检验结果：")
    st.text(f"  平均相对误差: {mean_relative_error:.4%}")
    st.text(f"  平均级比偏差: {mean_ratio_deviation:.4f}")
    # 给出一些参考值，根据你的具体应用场景调整
    if mean_relative_error < 0.1 and mean_ratio_deviation < 0.1:
      st.text("  模型精度较高")
    elif mean_relative_error < 0.2 and mean_ratio_deviation < 0.2:
      st.text("  模型精度尚可")
    else:
      st.text("  模型精度较低，可能需要考虑其他模型或调整参数")

    # 10. 构建结果 DataFrame
    if isinstance(data, pd.Series):
        index = data.index
    else:
        index = pd.RangeIndex(start=0, stop=len(data))

    result_df = pd.DataFrame({
        '原始值': data,
        '拟合值': fitted_values,
        '残差' : residuals,
        '相对误差': relative_errors
    }, index=index)

    forecast_index = pd.RangeIndex(start=len(data), stop=len(data) + forecast_steps)
    forecast_df = pd.DataFrame({
        '预测值': forecast_values
    }, index=forecast_index)

    result_df = pd.concat([result_df, forecast_df])

    # 11. 返回结果
    return result_df, a[0], b[0]

def prophet_forecast(data, period=30, freq='D', seasonality_mode='additive',
                     growth='linear', changepoint_prior_scale=0.05,
                     seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
                     interval_width=0.8, include_history=True, **kwargs):
    """
    简洁且全面的 Prophet 预测模型封装函数

    :param data: 时间序列数据，DataFrame 格式，必须包含 'ds' (日期) 和 'y' (值) 两列。
    :param period: 预测的未来时间步数，默认为 30。
    :param freq: 数据频率，例如 'D' 表示每日，'M' 表示每月，'Y' 表示每年，默认为 'D'。
                 也支持 Pandas 的时间频率别名，如 '15min', 'H', '7D' 等。
    :param seasonality_mode: 季节性模式，'additive' 或 'multiplicative'，默认为 'additive'。
    :param growth: 趋势增长模型，'linear' 或 'logistic'，默认为 'linear'。
                   如果是 'logistic'，数据中需要包含 'cap' 列来指定容量上限。
    :param changepoint_prior_scale: 趋势变化点的灵活性，越大越灵活，默认为 0.05。
    :param seasonality_prior_scale: 季节性强度的调节参数，越大季节性影响越强，默认为 10.0。
    :param holidays_prior_scale: 节假日效应的强度调节参数，默认为 10.0。
    :param interval_width: 预测区间宽度，默认为 0.8，表示 80% 的置信区间。
    :param include_history: 预测结果是否包含历史数据，默认为 True。
    :param kwargs: 其他 Prophet 模型的参数，例如 holidays, mcmc_samples 等。
    :return: 包含预测结果的 DataFrame，包括日期 (ds)、预测值 (yhat)、
             预测下限 (yhat_lower)、预测上限 (yhat_upper) 等列。
    """

    # 1. 数据检查和预处理
    if not isinstance(data, pd.DataFrame):
        raise TypeError("输入数据必须是 Pandas DataFrame 格式")
    if not {'ds', 'y'}.issubset(data.columns):
        raise ValueError("输入数据必须包含 'ds' (日期) 和 'y' (值) 两列")

    # 如果增长模式是 'logistic'，则需要 'cap' 列
    if growth == 'logistic':
        if 'cap' not in data.columns:
            raise ValueError("当 growth='logistic' 时，数据中需要包含 'cap' 列来指定容量上限")
        data['cap'] = data['cap'].astype(float) # 确保 cap 列是数值类型
    # 将 'ds' 列转换为日期时间类型
    data['ds'] = pd.to_datetime(data['ds'])

    # 2. 创建 Prophet 模型
    model = Prophet(
        growth=growth,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        interval_width=interval_width,
        **kwargs
    )

    # 3. 拟合模型
    model.fit(data)

    # 4. 构建未来日期框架
    future = model.make_future_dataframe(periods=period, freq=freq, include_history=include_history)

    # 如果增长模式是 'logistic'，未来日期框架也需要 'cap' 列
    if growth == 'logistic':
        if include_history:
          last_cap_index = data['ds'].last_valid_index()
          future['cap'] = data['cap'][last_cap_index]
        else:
          future['cap'] = data['cap'].iloc[-1] # 使用数据中最后一个 'cap' 值

    # 5. 进行预测
    forecast = model.predict(future)

    # 6. 返回预测结果
    return forecast

# ================= 页面配置 =================
st.set_page_config(layout="wide", page_title="时间序列预测")
st.title("时间序列预测分析")

# ================= 数据配置 =================
with st.expander("📊 数据配置", expanded=True):
    data_col1, data_col2 = st.columns([1, 2])

    with data_col1:
        st.subheader("示例数据生成")
        time_freq = st.selectbox("时间频率", ["D", "M", "Y"], index=0)
        n_samples = st.number_input("样本数量", 50, 1000, 365)
        add_noise = st.checkbox("添加噪声")
        add_outliers = st.checkbox("添加离群点")

        if st.button("生成示例数据"):
            dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=time_freq)
            base = np.linspace(50, 100, n_samples)
            y = base + 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))

            if add_noise:
                y += np.random.normal(0, 3, n_samples)
            if add_outliers:
                outlier_idx = np.random.choice(n_samples, size=max(3, n_samples // 20), replace=False)
                y[outlier_idx] *= 2.5

            # 确保非负（适用于灰色预测）
            y = np.abs(y)

            time_series_data = pd.DataFrame({
                'ds': dates,
                'y': y
            })

            st.session_state.update({
                "time_series_data": time_series_data,
                "time_series_use_example": True
            })
            st.success("示例数据生成成功!")

    with data_col2:
        st.subheader("上传自定义数据")
        uploaded_file = st.file_uploader("上传CSV/Excel文件", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # 自动检测时间列
                time_col = st.selectbox("选择时间列", df.columns)
                value_col = st.selectbox("选择数值列", df.columns)

                if st.button("确认数据格式"):
                    df['ds'] = pd.to_datetime(df[time_col])
                    df['y'] = df[value_col].astype(float)
                    df = df[['ds', 'y']].dropna()

                    st.session_state.update({
                        "time_series_data": df,
                        "time_series_use_example": False
                    })
                    st.success("数据加载成功!")

            except Exception as e:
                st.error(f"数据加载错误: {str(e)}")

# ================= 数据预览 =================
if "time_series_data" in st.session_state:
    with st.expander("🔍 数据预览", expanded=True):
        cols = st.columns(2)
        df = st.session_state.time_series_data
        cols[0].write(f"数据维度：{df.shape}")
        cols[0].dataframe(df)
        cols[1].line_chart(df.set_index('ds')['y'], use_container_width=True)

# ================= 方法配置 =================
st.markdown("---")
method = st.selectbox("选择预测方法",
                      ["ARIMA", "灰色预测", "Prophet"],
                      index=0)

params = {}
if method == "ARIMA":
    with st.expander("⚙️ ARIMA参数配置"):
        cols = st.columns(3)
        params['p_max'] = cols[0].number_input("p最大值", 0, 10, 2)
        params['d'] = cols[1].number_input("d值", 0, 3, 1)
        params['q_max'] = cols[2].number_input("q最大值", 0, 10, 2)
        params['forecast_num'] = st.number_input("预测步数", 1, 365, 30)

elif method == "灰色预测":
    with st.expander("⚙️ 灰色预测参数配置"):
        cols = st.columns(2)
        params['forecast_steps'] = cols[0].number_input("预测步数", 1, 100, 5)
        params['alpha'] = cols[1].number_input("紧邻均值系数", 0.0, 1.0, 0.5)

elif method == "Prophet":
    with st.expander("⚙️ Prophet参数配置"):
        cols = st.columns(2)
        params['period'] = cols[0].number_input("预测周期", 1, 365, 30)
        params['seasonality_mode'] = cols[1].selectbox("季节模式", ["additive", "multiplicative"])
        params['growth'] = st.selectbox("增长类型", ["linear", "logistic"])

# ================= 执行预测 =================
if st.button("开始预测") and "time_series_data" in st.session_state:
    df = st.session_state.time_series_data
    results = {}

    try:
        if method == "ARIMA":
            data_series = df.set_index('ds')['y']
            forecast_df = arima_forecast(data_series, **params)
            forecast_df.drop('forecast_se', axis=1, inplace=True)

        elif method == "灰色预测":
            result_df, a, b = grey_prediction(df['y'].values, **params)
            forecast_df = result_df[['预测值']].iloc[-params['forecast_steps']:]
            st.text(f"\n发展系数 a: {a:.4f}")
            st.text(f"灰作用量 b: {b:.4f}")

        elif method == "Prophet":
            if params['growth'] == 'logistic':
                df['cap'] = df['y'].max() * 1.2
            forecast_df = prophet_forecast(df, **params)
            forecast_df = forecast_df.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(params['period'])

        results[method] = forecast_df
        st.session_state.time_series_results = results
        st.success("预测完成!")

    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# ================= 结果展示 =================
if "time_series_results" in st.session_state:
    st.markdown("---")
    with st.expander("📈 预测结果", expanded=True):
        method = list(st.session_state.time_series_results.keys())[0]
        forecast_df = st.session_state.time_series_results[method]

        # 数据可视化
        col1, col2 = st.columns([2, 1])
        with col1:
            show_raw = st.checkbox("显示原始数据", value=True)
            chart_data = forecast_df.copy()
            if show_raw:
                raw_data = st.session_state.time_series_data.set_index('ds')['y']
                chart_data = pd.concat([raw_data, chart_data], axis=1)

            st.line_chart(chart_data, use_container_width=True)

        with col2:
            st.dataframe(forecast_df)

            # 结果下载
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                forecast_df.to_excel(writer, sheet_name='预测结果')
            st.download_button(
                label="下载预测结果",
                data=excel_buffer.getvalue(),
                file_name=f"{method}_预测结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.markdown("---")
st.caption("提示：灰色预测要求数据非负，Prophet需要包含'ds'和'y'列的时间序列数据")