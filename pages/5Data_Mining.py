import os
os.environ['OMP_NUM_THREADS'] = '1'
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import cohen_kappa_score
from scipy.cluster.hierarchy import linkage
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['font.family']='sans-serif'  # 设置字体样式
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# ================= 目标代码中的函数（保持原样） =================

def pca_reduction(data, n_components=None):
    """
    使用PCA方法进行降维
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param n_components: int，要降至的维数，默认为None(自动选择保留95%方差的维数)
    :return: numpy数组，降维后的数据
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_scaled)


def fisher_lda(data, labels, n_components=2):
    """
    使用Fisher线性判别分析进行降维
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param labels: numpy数组，数据标签据标签
    :param n_components: int，要降至的维数
    :return: numpy数组，降维后的数据
    """
    classes = np.unique(labels)
    total_mean = np.mean(data, axis=0)

    # 计算类内散度矩阵和类间散度矩阵
    Sw = np.zeros((data.shape[1], data.shape[1]))
    Sb = np.zeros((data.shape[1], data.shape[1]))

    for c in classes:
        class_data = data[labels == c]
        if len(class_data) < 1:
            continue  # 跳过空类别
        class_mean = np.mean(class_data, axis=0)
        Sw += np.dot((class_data - class_mean).T, (class_data - class_mean))
        diff = (class_mean - total_mean).reshape(-1, 1)
        Sb += len(class_data) * np.dot(diff, diff.T)

    # 添加正则化项
    Sw += 1e-6 * np.eye(Sw.shape[0])

    # 使用伪逆并提取实部
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    eiglist = sorted(zip(eigvals, eigvecs.T), key=lambda x: x[0], reverse=True)
    W = np.hstack([eigvec.reshape(-1, 1) for eigval, eigvec in eiglist[:n_components]])
    return np.dot(data, W)


def pearson_correlation(x, y):
    """
    计算两个变量的皮尔逊相关系数
    :param x: numpy数组或列表，第一个变量
    :param y: numpy数组或列表，第二个变量
    :return: float，相关系数和p值
    """
    return pearsonr(x, y)


def spearman_correlation(x, y):
    """
    计算两个变量的斯皮尔曼秩相关系数
    :param x: numpy数组或列表，第一个变量
    :param y: numpy数组或列表，第二个变量
    :return: float，相关系数和p值
    """
    return spearmanr(x, y)


def kappa_coefficient(rater1, rater2):
    """
    计算两个评分者之间的Kappa一致性系数
    :param rater1: numpy数组，第一个评分者的评分
    :param rater2: numpy数组，第二个评分者的评分
    :return: float，Kappa系数
    """
    return cohen_kappa_score(rater1, rater2)


def kmeans_clustering(data, n_clusters=3, random_state=42):
    """
    使用K-means算法进行聚类
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param n_clusters: int，聚类的数量
    :param random_state: int，随机种子
    :return: numpy数组，聚类标签
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit_predict(data)


def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    使用DBSCAN算法进行聚类
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param eps: float，邻域半径
    :param min_samples: int，成为核心点所需的最小样本数
    :return: numpy数组，聚类标签
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)


def hierarchical_clustering(data, method='ward'):
    """
    使用层次聚类算法进行聚类
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param method: str，聚类方法，可选'ward'，'complete'，'average'，'single'
    :return: numpy数组，层次聚类的连接矩阵
    """
    return linkage(data, method=method)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

@st.fragment()
def autoencoder_reduction(data, encoding_dim=2, epochs=50, batch_size=32, learning_rate=1e-3):
    """
    使用PyTorch实现的自编码器进行降维
    :param data: numpy数组，shape为(n_samples, n_features_X)的输入数据
    :param encoding_dim: int，要降至的维数
    :param epochs: int，训练轮数
    :param batch_size: int，批次大小
    :param learning_rate: float，学习率
    :return: numpy数组，降维后的数据
    """
    # 转换数据为torch张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.FloatTensor(data).to(device)
    input_dim = data.shape[1]

    # 初始化模型、优化器和损失函数
    model = Autoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 创建数据加载器
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

    # 获取编码后的数据
    model.eval()
    with torch.no_grad():
        encoded_data = model.encode(data_tensor)

    return encoded_data.cpu().numpy()


# ================= 页面配置 =================
st.set_page_config(layout="wide", page_title="🔎 数据挖掘分析")
st.title("🔎 数据挖掘分析")

# ================= 数据配置 =================
with st.expander("📊 数据配置", expanded=True):
    data_source = st.radio(
        "请选择数据来源",
        options=["生成示例数据", "上传自定义数据"],
        horizontal=True,
        key="data_mining_data_source_radio"
    )

    if data_source == "生成示例数据":
        st.subheader("✨ 示例数据生成")
        x1, x2 = st.columns(2)
        method_type = st.session_state.get("data_mining_method", "降维分析")
        n_features = x1.number_input("特征数量", 2, 50, 10, key="n_features")
        n_samples = x2.number_input("样本数量", 50, 1000, 200, key="n_samples")
        add_noise = x1.checkbox("添加噪声", key="add_noise")
        add_outliers = x2.checkbox("添加离群点", key="add_outliers")

        with st.form("data_mining_example_form"):
            submit_example = st.form_submit_button("生成示例数据")
            if submit_example:
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                if add_noise:
                    X += np.random.normal(0, 0.5, X.shape)
                if add_outliers:
                    outlier_idx = np.random.choice(n_samples, size=max(3, n_samples // 20), replace=False)
                    X[outlier_idx] *= 5
                y = np.random.randint(0, 3, n_samples)
                df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
                df["Label"] = y
                st.session_state.data_mining_data = df
                st.success("示例数据生成成功！")

    elif data_source == "上传自定义数据":
        st.subheader("📤 上传自定义数据")
        with st.form("data_mining_upload_form"):
            uploaded_file = st.file_uploader("上传CSV/Excel文件", type=["csv", "xlsx"], key="data_mining_upload")
            submit_upload = st.form_submit_button("上传数据")
            if submit_upload and uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    label_col = st.selectbox("选择标签列（可选）", ["无"] + list(df.columns), key="data_mining_label_col")
                    if label_col != "无":
                        df = df.rename(columns={label_col: "Label"})
                    st.session_state.data_mining_data = df
                    st.success("数据加载成功！")
                except Exception as e:
                    st.error(f"数据加载错误: {str(e)}")
            elif submit_upload and not uploaded_file:
                st.warning("请先上传文件。")

# ================= 数据预览 =================
with st.expander("🔍 数据预览", expanded=False):
    if "data_mining_data" not in st.session_state:
        st.warning("请先生成或上传数据")
        st.stop()
    df = st.session_state.data_mining_data
    cols = st.columns([2, 1])
    cols[0].dataframe(df)
    cols[1].write(f"📏 数据维度：{df.shape}")

# ================= 方法选择与参数配置 =================
st.markdown("---")
method_options = {
    "降维分析": ["PCA", "Fisher LDA", "自编码器"],
    "相关性分析": ["皮尔逊相关系数", "斯皮尔曼相关系数"],
    "一致性分析": ["Kappa系数"],
    "聚类分析": ["K-means", "DBSCAN", "层次聚类"]
}

method_category = st.selectbox("🧮 选择方法类别", list(method_options.keys()))
method = st.selectbox("🧩 选择具体方法", method_options[method_category])

params = {}
selected_features = []  # 新增：存储选择的特征
with st.expander("⚙️ 参数配置"):
    if method == "PCA":
        params["n_components"] = st.number_input("降维维度", 1, 10, 2)
    elif method == "Fisher LDA":
        params["n_components"] = st.number_input("降维维度", 1, df.shape[1]-1, 2)
    elif method == "自编码器":
        params["encoding_dim"] = st.number_input("编码维度", 1, 10, 2)
    elif method == "Kappa系数":
        col1, col2 = st.columns(2)
        rater1 = col1.selectbox("选择评分者1", df.columns)
        rater2 = col2.selectbox("选择评分者2", df.columns)

        # 新增：连续数据离散化处理
        if np.issubdtype(df[rater1].dtype, np.number) or np.issubdtype(df[rater2].dtype, np.number):
            st.warning("检测到连续型数据，Kappa分析需要离散值，请设置分箱参数")

            bin_method = st.selectbox("分箱方法", ["等宽分箱", "等频分箱"])
            n_bins = st.number_input("分箱数量", 2, 10, 3)
            params["bin_method"] = bin_method
            params["n_bins"] = n_bins

            if bin_method == "等宽分箱":
                rater1_binned = pd.cut(df[rater1], bins=n_bins, labels=False)
                rater2_binned = pd.cut(df[rater2], bins=n_bins, labels=False)
            else:
                rater1_binned = pd.qcut(df[rater1], q=n_bins, labels=False, duplicates='drop')
                rater2_binned = pd.qcut(df[rater2], q=n_bins, labels=False, duplicates='drop')

            params["rater1"] = rater1_binned
            params["rater2"] = rater2_binned

        else:
            params["rater1"] = df[rater1]
            params["rater2"] = df[rater2]

    elif method in ["K-means"]:
        params["n_clusters"] = st.number_input("聚类数量", 2, 10, 3)
    elif method in [ "层次聚类"]:
        params["n_clusters"] = st.number_input("聚类数量", 2, 10, 3)
        params["ccway"] = st.selectbox("选择方法",['ward','complete','average','single'])
    elif method == "DBSCAN":
        params["eps"] = st.number_input("邻域半径", 0.1, 5.0, 0.5)
        params["min_samples"] = st.number_input("最小样本数", 1, 20, 5)
    # 新增：相关性分析特征选择
    elif method in ["皮尔逊相关系数", "斯皮尔曼相关系数"]:
        available_features = df.columns.tolist()
        if "Label" in available_features:
            available_features.remove("Label")
        selected_features = st.multiselect(
            "选择分析特征（可多选）",
            available_features,
            default=available_features[:2] if len(available_features)>=2 else []
        )
# ================= 执行分析 =================
if st.button("🚀 开始分析") and "data_mining_data" in st.session_state:
    df = st.session_state.data_mining_data
    results = {}

    try:
        if method == "PCA":
            X = df.drop(columns=["Label"]) if "Label" in df.columns else df
            reduced_data = pca_reduction(X.values, **params)
            results["result"] = pd.DataFrame(reduced_data, columns=[f"PC_{i}" for i in range(reduced_data.shape[1])])

        elif method == "Fisher LDA":
            X = df.drop(columns=["Label"]).values
            y = df["Label"].values
            reduced_data = fisher_lda(X, y, **params)
            results["result"] = pd.DataFrame(reduced_data, columns=[f"LD_{i}" for i in range(reduced_data.shape[1])])

        elif method == "自编码器":
            X = StandardScaler().fit_transform(df.drop(columns=["Label"]) if "Label" in df.columns else df)
            reduced_data = autoencoder_reduction(X, **params)
            results["result"] = pd.DataFrame(reduced_data, columns=[f"AE_{i}" for i in range(reduced_data.shape[1])])


        elif method in ["皮尔逊相关系数", "斯皮尔曼相关系数"]:
            corr_func = pearson_correlation if method == "皮尔逊相关系数" else spearman_correlation
            # 新增：多特征处理
            if len(selected_features) < 2:
                st.error("请至少选择两个特征进行分析")
                st.stop()
            # 计算相关系数矩阵
            corr_matrix = np.zeros((len(selected_features), len(selected_features)))
            p_matrix = np.zeros((len(selected_features), len(selected_features)))
            for i, feat1 in enumerate(selected_features):
                for j, feat2 in enumerate(selected_features):
                    corr, p = corr_func(df[feat1], df[feat2])
                    corr_matrix[i][j] = corr
                    p_matrix[i][j] = p

            # 创建完整的相关系数矩阵DataFrame
            corr_df = pd.DataFrame(
                corr_matrix,
                index=selected_features,
                columns=selected_features
            )
            results["result"] = corr_df  # 确保result字段始终有值
            results["corr_matrix"] = corr_matrix
            results["p_matrix"] = p_matrix
            results["selected_features"] = selected_features
            # 保留原有双特征显示
            if len(selected_features) == 2:
                results["result"] = pd.DataFrame([[corr_matrix[0][1], p_matrix[0][1]]],columns=["相关系数", "p值"])

        elif method == "Kappa系数":
            if np.issubdtype(df[rater1].dtype, np.number) or np.issubdtype(df[rater2].dtype, np.number):
                kappa = kappa_coefficient(params["rater1"], params["rater2"])
                results["result"] = pd.DataFrame([[kappa, params["bin_method"], params["n_bins"]]],
                                                 columns=["Kappa系数", "分箱方法", "分箱数量"])
            else:
                kappa = kappa_coefficient(params["rater1"], params["rater2"])
                results["result"] = pd.DataFrame([[kappa]], columns=["Kappa系数"])

        elif method == "K-means":
            X = StandardScaler().fit_transform(df)
            labels = kmeans_clustering(X, **params)
            results["result"] = pd.DataFrame(labels, columns=["Cluster"])

        elif method == "DBSCAN":
            X = StandardScaler().fit_transform(df)
            labels = dbscan_clustering(X, **params)
            results["result"] = pd.DataFrame(labels, columns=["Cluster"])

        elif method == "层次聚类":
            X = StandardScaler().fit_transform(df)
            linkage_matrix = hierarchical_clustering(X,params["ccway"])
            results["result"] = pd.DataFrame(linkage_matrix)

        st.session_state.data_mining_results = results
        st.success("分析完成！")

    except Exception as e:
        st.error(f"分析失败: {str(e)}")

# ================= 结果展示 =================
if "data_mining_results" in st.session_state:
    st.markdown("---")
    with st.expander("📈 分析结果", expanded=True):
        result_df = st.session_state.data_mining_results.get("result")
        corr_matrix = st.session_state.data_mining_results.get("corr_matrix")
        selected_features = st.session_state.data_mining_results.get("selected_features")
        linkage_matrix = st.session_state.data_mining_results.get("linkage_matrix")

        # 可视化展示重构
        col1, col2 = st.columns([2, 1])
        with col1:
            if method in ["PCA", "Fisher LDA", "自编码器"]:
                st.subheader("降维可视化(仅前两个维度)")
                if result_df.shape[1] >= 2:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        x=result_df.iloc[:, 0],
                        y=result_df.iloc[:, 1],
                        hue=df["Label"] if "Label" in df.columns else None,
                        palette="viridis"
                    )
                    plt.xlabel(result_df.columns[0])
                    plt.ylabel(result_df.columns[1])
                    plt.title(f"{method} 降维分布")
                    st.pyplot(plt)
                else:
                    st.line_chart(result_df)

            elif method in ["皮尔逊相关系数", "斯皮尔曼相关系数"]:
                if len(selected_features) == 2:
                    st.subheader("特征散点图")
                    sns.jointplot(
                        data=df,
                        x=selected_features[0],
                        y=selected_features[1],
                        kind="reg"
                    )
                    st.pyplot(plt)
                else:
                    st.subheader("相关系数矩阵")
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        xticklabels=selected_features,
                        yticklabels=selected_features
                    )
                    st.pyplot(plt)

            elif method in ["K-means", "DBSCAN"]:
                st.subheader("聚类分布")
                if result_df.shape[0] == df.shape[0]:
                    plot_df = pd.concat([
                        df.drop(columns=["Label"], errors="ignore"),
                        result_df
                    ], axis=1)
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        data=plot_df,
                        x=plot_df.columns[0],
                        y=plot_df.columns[1],
                        hue="Cluster",
                        palette="tab10"
                    )
                    st.pyplot(plt)

            elif method == "层次聚类":
                st.subheader("树状图")
                plt.figure(figsize=(12, 6))
                dendrogram = sns.clustermap(
                    df.drop(columns=["Label"], errors="ignore"),
                    row_linkage=linkage_matrix,
                    col_cluster=False
                )
                st.pyplot(dendrogram.fig)

        # 下载功能增强
        with col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # 通用结果保存
                result_df.to_excel(writer, sheet_name='分析结果')

                # 方法特定数据保存
                if method in ["皮尔逊相关系数", "斯皮尔曼相关系数"]:
                    pd.DataFrame(
                        corr_matrix,
                        columns=selected_features,
                        index=selected_features
                    ).to_excel(writer, sheet_name='相关系数矩阵')

                    pd.DataFrame(
                        st.session_state.data_mining_results.get("p_matrix"),
                        columns=selected_features,
                        index=selected_features
                    ).to_excel(writer, sheet_name='P值矩阵')

                if method == "层次聚类":
                    pd.DataFrame(
                        linkage_matrix,
                        columns=['Cluster1', 'Cluster2', 'Distance', 'SampleCount']
                    ).to_excel(writer, sheet_name='连接矩阵')

                if method in ["K-means", "DBSCAN"]:
                    cluster_profile = df.groupby(result_df['Cluster']).mean()
                    cluster_profile.to_excel(writer, sheet_name='聚类中心特征')

            # 显示精简结果
            st.subheader("⭐ 核心指标")
            if method == "Kappa系数":
                st.metric(label="Kappa系数", value=f"{result_df.iloc[0, 0]:.3f}")
            elif method in ["皮尔逊相关系数", "斯皮尔曼相关系数"] and len(selected_features) == 2:
                st.metric(label="相关系数", value=f"{result_df.iloc[0, 0]:.3f}")
                st.metric(label="P值", value=f"{result_df.iloc[0, 1]:.4f}")
            else:
                st.dataframe(result_df)
            st.download_button(
                label="⬇️ 下载完整分析结果",
                data=excel_buffer.getvalue(),
                file_name=f"{method}_分析结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )