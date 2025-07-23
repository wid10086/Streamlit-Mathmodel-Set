import streamlit as st
import numpy as np
import pandas as pd

# ================= 工具方法定义（保持原有逻辑不变） =================
# [原有ahp(), topsis(), fuzzy_evaluation(), entropy_weight_method()定义...]

def ahp(criteria_matrix, alternative_matrices=None):
    """
    层次分析法 (Analytic Hierarchy Process, AHP)

    :param criteria_matrix: 准则层判断矩阵，二维数组
    :param alternative_matrices: 方案层判断矩阵列表，每个元素是一个二维数组，对应一个准则
    :return: 如果提供了方案层判断矩阵，则返回每个方案的最终得分；否则返回准则层权重
    """

    # 检查判断矩阵是否为方阵且对角线元素为1
    def check_matrix(matrix):
        if matrix.shape[0] != matrix.shape[1]:
            st.warning("判断矩阵必须是方阵")
            st.stop()
        if not np.allclose(np.diag(matrix), 1):
            st.warning("判断矩阵对角线元素必须为1")
            st.stop()

    check_matrix(criteria_matrix)

    # 计算准则层权重
    eigenvalues, eigenvectors = np.linalg.eig(criteria_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    criteria_weights = np.real(eigenvectors[:, max_eigenvalue_index] / np.sum(eigenvectors[:, max_eigenvalue_index]))

    # 一致性检验
    n = criteria_matrix.shape[0]
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]  # 随机一致性指标
    CI = (eigenvalues[max_eigenvalue_index] - n) / (n - 1)
    CR = CI / RI[n - 1] if n > 2 else 0 # 当n=1,2时，总是完全一致的

    if CR >= 0.1:
        st.text("警告：准则层一致性检验未通过 (CR >= 0.1)")

    if alternative_matrices is None:
        st.text(f"准则层一致性比率 CR: {CR:.4f}")
        return criteria_weights

    # 计算方案层权重
    alternative_weights = []
    for i, alternative_matrix in enumerate(alternative_matrices):
        check_matrix(alternative_matrix)

        eigenvalues, eigenvectors = np.linalg.eig(alternative_matrix)
        max_eigenvalue_index = np.argmax(eigenvalues)
        weights = np.real(eigenvectors[:, max_eigenvalue_index] / np.sum(eigenvectors[:, max_eigenvalue_index]))

        # 一致性检验
        n_alt = alternative_matrix.shape[0]
        CI_alt = (eigenvalues[max_eigenvalue_index] - n_alt) / (n_alt - 1)
        CR_alt = CI_alt / RI[n_alt - 1] if n_alt > 2 else 0

        st.text(f"方案层（准则 {i+1}）一致性比率 CR: {CR_alt:.4f}")
        if CR_alt >= 0.1:
          st.text(f"警告：方案层（准则 {i+1}）一致性检验未通过 (CR >= 0.1)")

        alternative_weights.append(weights)

    alternative_weights = np.array(alternative_weights)

    # 计算最终得分
    final_scores = np.dot(criteria_weights, alternative_weights)

    return final_scores

def topsis(data_matrix, weights, benefit_attributes):
    """
    TOPSIS 法 (Technique for Order Preference by Similarity to an Ideal Solution)

    :param data_matrix: 决策矩阵，二维数组，每一行是一个方案，每一列是一个属性
    :param weights: 属性权重，一维数组
    :param benefit_attributes: 效益属性（越大越好）的索引列表，例如 [0, 2] 表示第1个和第3个属性是效益属性
    :return: 每个方案的相对接近度
    """
    data_matrix = np.asarray(data_matrix)
    weights = np.asarray(weights)
    # 新增权重归一化
    weights = weights / np.sum(weights)
    # 1. 规范化决策矩阵
    normalized_matrix = data_matrix / np.linalg.norm(data_matrix, axis=0)

    # 2. 加权规范化决策矩阵
    weighted_matrix = normalized_matrix * weights

    # 3. 确定正理想解和负理想解
    positive_ideal_solution = np.zeros(data_matrix.shape[1])
    negative_ideal_solution = np.zeros(data_matrix.shape[1])
    for i in range(data_matrix.shape[1]):
        if i in benefit_attributes:
            positive_ideal_solution[i] = np.max(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.min(weighted_matrix[:, i])
        else:
            positive_ideal_solution[i] = np.min(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.max(weighted_matrix[:, i])

    # 4. 计算每个方案到正理想解和负理想解的距离
    distance_to_positive = np.linalg.norm(weighted_matrix - positive_ideal_solution, axis=1)
    distance_to_negative = np.linalg.norm(weighted_matrix - negative_ideal_solution, axis=1)

    # 5. 计算每个方案的相对接近度
    closeness = distance_to_negative / (distance_to_positive + distance_to_negative + 1e-8)

    return closeness

def fuzzy_evaluation(criteria_weights, membership_degrees, evaluation_levels):
    """
    模糊综合评价法 (Fuzzy Comprehensive Evaluation)

    :param criteria_weights: 准则权重, 一维数组
    :param membership_degrees: 各方案在各准则下的隶属度矩阵, 三维数组,
                               第一维为方案, 第二维为准则, 第三维为评语等级
    :param evaluation_levels: 评语等级权重, 一维数组
    :return: 各方案的模糊综合评价得分
    """
    criteria_weights = np.asarray(criteria_weights)
    membership_degrees = np.asarray(membership_degrees)
    evaluation_levels = np.asarray(evaluation_levels)

    # 处理二维输入（单方案情况）
    if membership_degrees.ndim == 2:
        membership_degrees = membership_degrees[np.newaxis, :, :]

    # 检查维度是否匹配
    if membership_degrees.shape[1] != criteria_weights.shape[0]:
        st.warning("隶属度矩阵的准则维度与准则权重维度不匹配")
        st.stop()
    if membership_degrees.shape[2] != evaluation_levels.shape[0]:
        st.warning("隶属度矩阵的评语等级维度与评语等级权重维度不匹配")
        st.stop()

    # 归一化准则权重
    criteria_weights = criteria_weights / criteria_weights.sum()

    # 计算模糊综合评价矩阵
    evaluation_matrix = np.dot(criteria_weights, membership_degrees)

    # 处理可能的除零错误
    sum_eval = np.sum(evaluation_matrix, axis=1, keepdims=True)
    if np.any(sum_eval == 0):
        st.warning("存在方案的合成隶属度总和为零，可能导致结果异常")
        sum_eval[sum_eval == 0] = 1  # 避免除零

    evaluation_matrix = evaluation_matrix / sum_eval

    # 计算最终得分
    final_scores = np.dot(evaluation_matrix, evaluation_levels)

    return final_scores

def entropy_weight_method(data_matrix):
    """
    熵权法 (Entropy Weight Method) 仅适用于那些变异程度能够反映指标重要程度的情况

    :param data_matrix: 决策矩阵，二维数组，每一行是一个方案，每一列是一个属性
    :return: 各个属性的权重
    """
    data_matrix = np.asarray(data_matrix)

    # 1. 规范化决策矩阵
    normalized_matrix = data_matrix / np.sum(data_matrix, axis=0)

    # 2. 计算每个属性的信息熵
    k = -1 / np.log(data_matrix.shape[0])
    entropy = np.zeros(data_matrix.shape[1])
    for j in range(data_matrix.shape[1]):
      entropy[j] = k * np.sum(normalized_matrix[:, j] * np.log(normalized_matrix[:, j] + 1e-10)) # 避免 log(0) 错误

    # 3. 计算每个属性的熵权
    weights = (1 - entropy) / np.sum(1 - entropy)

    return weights

# ================= 页面配置 =================
st.set_page_config(layout="wide")
st.title("📊 多准则决策分析/评估")
criteria_names = []
ways_name = []
st.session_state.eval_data = None

# ================= 侧边栏 - 方法选择 =================
with st.container(border= True):
    st.markdown("### ⚙️ 配置")
    col1, col2 = st.columns(2)
    with col1:
        n_features = st.number_input("🔢 特征/准则数量", min_value=2, max_value=10, value=3, key="evaluate_n_features")
        way_num = st.number_input("🧑‍🤝‍🧑 方案数", min_value=1, max_value=10, value=1, key="way_num")
        method = st.selectbox(
            "🧮 选择分析方法",
            ["AHP", "TOPSIS", "模糊综合评价", "熵权法"],
            key="evaluate_method"
        )

    with col2:
        with st.expander("📝 自定义准则名称(不可重名)"):
            for i in range(n_features):
                criteria_names.append(st.text_input(f"准则{i + 1}", value=f"C{i + 1}"))
        with st.expander("📝 自定义方案名称(不可重名)"):
            for i in range(way_num):
                ways_name.append(st.text_input(f"方案{i + 1}", value=f"P{i + 1}"))
        if method =="TOPSIS" or method == "熵权法":
            with st.expander("📤 自定义文件上传"):
                with st.form("文件上传"):
                    file = st.file_uploader(f"依照示例格式上传数据文件（第一列为数据，不是列名）,", type=["xlsx", "xls"])
                    sheet = st.text_input(f"数据工作表名（Excel）", value="Sheet1")
                    if st.form_submit_button("数据上传"):
                        if file:
                            data = pd.read_excel(file,sheet_name=sheet)
                            criteria_names = data.columns.tolist()
                            st.session_state.eval_data = data
                            ways_name = data.index.tolist()
                            n_features = len(criteria_names)
                            way_num = len(ways_name)
                        else:
                            st.text("无上传文件")
                            st.session_state.eval_data = None


# ================= 方法参数配置 =================
params = {}

@st.fragment()
def fun(method,n_features,way_num):
    with st.container(border=True):
        row = st.columns(3)
        row[0].markdown("#### ⚙️ 参数配置")
        character = criteria_names
        show = row[1].checkbox("👁️ 查看矩阵")
        if method == "AHP":
            row = st.columns(2)
            with row[0]:
                with st.expander("🟦 准则相对重要程度（1/9~9,只需填写右上方表格）", expanded=True):
                    st.markdown("填写判断矩阵，a(i,j)>1表示i比j更重要，a(i,j)<1表示i比j更不重要。主对角线必须为1。")
                    # 默认生成1的矩阵
                    matrix_df = pd.DataFrame(np.ones((n_features, n_features)), index=criteria_names, columns=criteria_names)
                    for i in range(n_features):
                        for j in range(i):
                            matrix_df.iloc[i, j] = None
                    # 允许用户编辑上三角（不含对角线），下三角自动填1/值
                    edited_df = st.data_editor(
                        matrix_df,
                        key="ahp_matrix_editor",
                        num_rows="fixed",
                        use_container_width=True
                    )
                    # 保证主对角线为1，下三角为上三角的倒数
                    for i in range(n_features):
                        edited_df.iloc[i, i] = 1.0
                        for j in range(i+1, n_features):
                            edited_df.iloc[j, i] = 1.0 / edited_df.iloc[i, j]
                    matrix = edited_df.values

                criteria_matrix = pd.DataFrame(matrix, index=criteria_names, columns=criteria_names)
                if show:
                    st.markdown("准则层判断矩阵")
                    st.dataframe(criteria_matrix, key="criteria")
            with row[1]:
                alternative_matrices = []
                with st.container( border=True):
                    st.markdown("#### 🟩 AHP方案层(得分规则同准则相对重要程度) ")
                    for i in range(n_features):
                        # 默认生成1的矩阵
                        alt_matrix_df = pd.DataFrame(np.ones((way_num, way_num)), index=ways_name, columns=ways_name)
                        for j in range(way_num):
                            for k in range(j):
                                alt_matrix_df.iloc[j, k] = None
                        with st.expander(f"{criteria_names[i]}中各方案相对得分（填写上三角，主对角线为1）"):
                            edited_alt_df = st.data_editor(
                                alt_matrix_df,
                                key=f"ahp_alt_matrix_editor_{i}",
                                num_rows="fixed",
                                use_container_width=True
                            )
                            # 保证主对角线为1，下三角为上三角的倒数
                            for j in range(way_num):
                                edited_alt_df.iloc[j, j] = 1.0
                                for k in range(j+1, way_num):
                                    edited_alt_df.iloc[k, j] = 1.0 / edited_alt_df.iloc[j, k]
                            x = edited_alt_df.values
                            if show:
                                st.dataframe(pd.DataFrame(x,index=ways_name,columns=ways_name),key=f"方案{i}")
                        alternative_matrices.append(x)
            params['criteria_matrix'] = criteria_matrix.values
            params['alternative_matrices'] = alternative_matrices

        elif method == "TOPSIS":
            data_matrix_topsis = np.ones((way_num, n_features))
            if st.session_state.eval_data is None:
                    # 构建默认数据表
                data_matrix_df = pd.DataFrame(
                    np.ones((way_num, n_features)),
                    index=ways_name,
                    columns=criteria_names
                )
                # 允许用户直接编辑整个表格
                edited_df = st.data_editor(
                    data_matrix_df,
                    key="topsis_data_matrix_editor" ,
                    num_rows="fixed",
                    use_container_width=True
                )
                data_matrix_topsis = edited_df.values 
            else:
                st.text("已存在文件")
                data_matrix_topsis = st.session_state.eval_data.values
            with st.expander("🟨 准则权重(会自动归一化)"):
                # 构建权重表格
                weights_df = pd.DataFrame(
                    np.ones((1, n_features)) / n_features,
                    columns=criteria_names,
                    index=["权重"]
                )
                edited_weights_df = st.data_editor(
                    weights_df,
                    key="topsis_weights_editor",
                    num_rows="fixed",
                    use_container_width=True
                )
                topsis_weights = edited_weights_df.values

                params["weights_topsis"] = topsis_weights
            params["benefit_attributes_topsis"] = st.multiselect(
                "✨ 选择效益属性（越大越好的特征,未选特征即为越大越坏的特征）",
                options=list(range(n_features)),
                format_func=lambda x: f"{criteria_names[x]}"
            )
            if show:
                st.text("决策矩阵")
                st.dataframe(pd.DataFrame(data_matrix_topsis,index=ways_name,columns=character),key="决策矩阵")
            params["data_matrix_topsis"] = data_matrix_topsis

        elif method == "模糊综合评价":
            cols=st.columns(2)
            with cols[0]:
                eval_num = st.number_input("🔢 评语等级数量",min_value=2,max_value=5,value=3)
                evaluation_levels_fuzzy = np.zeros(eval_num)
                str_k = ['优','较优','中','较差','差']
                if eval_num ==2:
                    str_eval = [str_k[0],str_k[4]]
                elif eval_num==3:
                    str_eval = [str_k[0],str_k[2],str_k[4]]
                elif eval_num==4:
                    str_eval = [str_k[0],str_k[2],str_k[3],str_k[4]]
                elif eval_num==5:
                    str_eval = str_k
                with st.expander("🟪 评语等级设置"):
                    col = st.columns(eval_num)
                    for i in range(eval_num):
                        level_value = col[i].number_input(
                            f"评语等级 {str_eval[i]} 的值",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0 - i *1.0/eval_num,  # 默认值递减
                            step=0.1,
                            key=f"level_{i}"
                        )
                        evaluation_levels_fuzzy[i] = level_value
                    # 验证评语等级值是否递减
                    is_valid = True
                    for i in range(1, eval_num):
                        if evaluation_levels_fuzzy[i] >= evaluation_levels_fuzzy[i - 1]:
                            is_valid = False
                            break
                    # 显示验证结果
                    if is_valid:
                        st.success("评语等级值设置正确（递减）")
                    else:
                        st.error("评语等级值设置错误：值必须是递减的！")
                criteria_weights_fuzzy = np.ones(n_features)
                with st.expander("🟨 准则权重(会自动归一化)"):
                    col = st.columns((n_features))
                    for i in range(n_features):
                        criteria_weights_fuzzy[i] = col[i].number_input(f"{criteria_names[i]}", key=f"准则权重{criteria_names[i]}",
                                                                    min_value=0.0, value=1.0 / n_features)

                params["criteria_weights_fuzzy"] = criteria_weights_fuzzy

                params["evaluation_levels_fuzzy"] = evaluation_levels_fuzzy
            with cols[1]:
                with st.container(border=True):
                    st.markdown("#### 🟧 模糊综合评价方案（直接在表格中填写各方案的隶属度）")
                    membership_degrees_fuzzy = []
                    for i in range(way_num):
                        # 构建与原来一致的默认隶属度矩阵
                        default_values = np.zeros((n_features, eval_num))
                        for j in range(n_features):
                            for k in range(eval_num):
                                default_values[j, k] = 1.0 - 1 * k / eval_num
                        fuzzy_df = pd.DataFrame(
                            default_values,
                            index=criteria_names,
                            columns=str_eval
                        )
                        # 允许用户直接编辑整个表格
                        edited_fuzzy_df = st.data_editor(
                            fuzzy_df,
                            key=f"fuzzy_membership_{i}",
                            num_rows="fixed",
                            use_container_width=True
                        )
                        a = edited_fuzzy_df.values
                        if show:
                            st.dataframe(pd.DataFrame(a, index=criteria_names, columns=str_eval), key=f"{ways_name[i]}隶属度")
                        membership_degrees_fuzzy.append(a)
                    params['membership_degrees_fuzzy'] = np.array(membership_degrees_fuzzy)
        elif method == "熵权法":
            if st.session_state.eval_data is None:
                with st.expander("🟦 不同方案在不同准则的属性"):
                    # 构建默认数据表
                    data_matrix_df = pd.DataFrame(
                        np.ones((way_num, n_features)),
                        index=ways_name,
                        columns=criteria_names
                    )
                    # 允许用户直接编辑整个表格
                    edited_df = st.data_editor(
                        data_matrix_df,
                        key="entropy_data_matrix_editor",
                        num_rows="fixed",
                        use_container_width=True
                    )
                    data_matrix_entropy = edited_df.values
            else:
                st.text("已存在文件")
                data_matrix_entropy = st.session_state.eval_data.values
            if show:
                st.text("决策矩阵")
                st.dataframe(pd.DataFrame(data_matrix_entropy, index=ways_name, columns=character), key="决策矩阵")
            params['data_matrix_entropy'] = data_matrix_entropy
        if st.button("🚀 开始计算"):
            #params
            st.markdown("---")
            if method == "AHP":
                criteria_weights = ahp(params['criteria_matrix'])
                st.write(f"层次分析法 - 准则层权重: {criteria_weights}")
                st.markdown("---")
                final_scores_ahp = ahp(params['criteria_matrix'], params['alternative_matrices'])
                st.markdown("---")
                row = st.columns(2)
                row[0].write(f"层次分析法 - 方案最终得分:")
                f = {}
                for i in range(way_num):
                    f[ways_name[i]]=final_scores_ahp[i]
                    row[0].write(f"{ways_name[i]}: {final_scores_ahp[i]}")
                row[1].bar_chart(f)
            elif method == "TOPSIS":
                closeness = topsis(params['data_matrix_topsis'], params['weights_topsis'], params['benefit_attributes_topsis'])
                row = st.columns(2)
                row[0].write(f"TOPSIS法 - 方案相对接近度:")
                f = {}
                for i in range(way_num):
                    f[ways_name[i]] = closeness[i]
                    row[0].write(f"{ways_name[i]}: {closeness[i]}")
                row[1].bar_chart(f)

            elif method == "模糊综合评价":
                final_scores_fuzzy = fuzzy_evaluation(params['criteria_weights_fuzzy'], params['membership_degrees_fuzzy'],
                                                      params['evaluation_levels_fuzzy'])
                row = st.columns(2)
                row[0].write(f"模糊综合评价法 - 方案最终得分:")
                f = {}
                for i in range(way_num):
                    f[ways_name[i]] = final_scores_fuzzy[i]
                    row[0].write(f"{ways_name[i]}: {final_scores_fuzzy[i]}")
                row[1].bar_chart(f)

            elif method == "熵权法":
                weights_entropy = entropy_weight_method(params['data_matrix_entropy'])
                st.write(f"熵权法 - 属性权重: {weights_entropy}")
                row = st.columns(2)
                row[0].write(f"熵权法 - 属性权重:")
                f = {}
                for i in range(n_features):
                    f[criteria_names[i]] = weights_entropy[i]
                    row[0].write(f"{criteria_names[i]}: {weights_entropy[i]}")
                row[1].bar_chart(f)
fun(method,n_features,way_num)
