import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
import random
import math
from scipy.spatial import distance  # 添加缺失的导入

def ant_colony_optimization(distance_matrix, n_ants=10, n_iterations=100, alpha=1, beta=2,
                            evaporation_rate=0.1, Q=1):
    """
    蚁群优化算法实现（适用于TSP问题）
    :param distance_matrix: 距离矩阵，表示城市间距离
    :param n_ants: 蚂蚁数量
    :param n_iterations: 迭代次数
    :param alpha: 信息素重要程度因子
    :param beta: 启发式因子重要程度因子
    :param evaporation_rate: 信息素蒸发率
    :param Q: 信息素增加强度系数
    :return: (best_path, best_length) 最优路径和其长度
    """
    n_cities = len(distance_matrix)
    # 增加矩阵有效性校验
    np.fill_diagonal(distance_matrix, 0)  # 确保对角线为0
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T)  # 确保对称性

    # 初始化信息素矩阵（增加最小值保护）
    pheromone = np.ones((n_cities, n_cities)) / n_cities + 1e-10

    # 优化启发式信息计算
    heuristic = 1 / (distance_matrix + np.eye(n_cities))  # 避免除零错误

    best_path = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        paths = []  # 存储本次迭代所有蚂蚁的路径
        path_lengths = []  # 存储本次迭代所有蚂蚁的路径长度

        # 每只蚂蚁构建解
        for ant in range(n_ants):
            path = [random.randint(0, n_cities - 1)]  # 随机选择起始城市
            while len(path) < n_cities:
                current = path[-1]
                unvisited = list(set(range(n_cities)) - set(path))

                # 计算转移概率
                probs = []
                for next_city in unvisited:
                    prob = (pheromone[current][next_city] ** alpha *
                            heuristic[current][next_city] ** beta)
                    probs.append(prob)

                # 轮盘赌选择下一个城市
                probs = np.array(probs) / sum(probs)
                next_city = np.random.choice(unvisited, p=probs)
                path.append(next_city)

            # 计算路径长度
            length = sum(distance_matrix[path[i]][path[i + 1]]
                         for i in range(n_cities - 1))
            length += distance_matrix[path[-1]][path[0]]  # 回到起点

            paths.append(path)
            path_lengths.append(length)

            # 更新最优解
            if length < best_length:
                best_length = length
                best_path = path[:]

        # 更新信息素
        pheromone *= (1 - evaporation_rate)  # 信息素蒸发
        for path, length in zip(paths, path_lengths):
            for i in range(n_cities - 1):
                pheromone[path[i]][path[i + 1]] += Q / length
            pheromone[path[-1]][path[0]] += Q / length

    return best_path, best_length

# ================= 算法选择与参数配置 =================
st.set_page_config(layout="wide")
st.title("启发式算法")

# ================= 问题配置 =================
with st.expander("问题配置", expanded=True):
    problem_type = st.radio("问题类型", ["函数优化", "TSP问题"], key="heuristic_problem_type")

# 动态算法选择
if problem_type == "TSP问题":
    algorithm_options = ["蚁群算法(TSP)"]
else:
    algorithm_options = ["遗传算法", "差分进化算法", "粒子群优化", "模拟退火算法"]

heuristic_method = st.selectbox(
    "选择优化算法",
    algorithm_options,
    key="heuristic_method"
)

# ================= 动态参数输入 =================
with st.expander("算法参数配置", expanded=True):
    col1, col2 = st.columns(2)

    # 公共参数
    with col1:
        max_iter = st.number_input("最大迭代次数", 10, 1000, 100, key="heuristic_max_iter")
        if heuristic_method != "蚁群算法(TSP)":
            dim = st.number_input("问题维度", 1, 10, 2, key="heuristic_dim")
        else:
            evaporation_rate = st.slider("蒸发率ρ", 0.01, 0.5, 0.1, key="heuristic_aco_evap")
            Q = st.number_input("信息素量Q", 0.1, 10.0, 1.0, key="heuristic_aco_Q")

    # 算法特定参数
    with col2:
        if heuristic_method == "遗传算法":
            pop_size = st.number_input("种群规模", 10, 500, 50, key="heuristic_ga_pop")
            mutation_rate = st.slider("变异概率", 0.0, 1.0, 0.1, key="heuristic_ga_mut")

        elif heuristic_method == "差分进化算法":
            size_pop = st.number_input("种群规模", 10, 500, 50, key="heuristic_de_pop")

        elif heuristic_method == "粒子群优化":
            pop = st.number_input("粒子数量", 10, 500, 40, key="heuristic_pso_pop")
            w = st.slider("惯性权重", 0.1, 1.5, 0.8, key="heuristic_pso_w")

        elif heuristic_method == "模拟退火算法":
            T_max = st.number_input("初始温度", 1, 1000, 100, key="heuristic_sa_Tmax")
            L = st.number_input("迭代次数/温度", 10, 1000, 300, key="heuristic_sa_L")


        elif heuristic_method == "蚁群算法(TSP)":
            n_ants = st.number_input("蚂蚁数量", 5, 100, 10, key="heuristic_aco_ants")
            alpha = st.slider("信息素权重α", 0.1, 5.0, 1.0, key="heuristic_aco_alpha")
            beta = st.slider("启发式权重β", 0.1, 5.0, 2.0, key="heuristic_aco_beta")

# ================= 问题配置 =================
with st.expander("问题配置", expanded=True):
    #problem_type = st.radio("问题类型", ["函数优化", "TSP问题"], key="heuristic_problem_type")

    if problem_type == "函数优化":
        func_input = st.text_area("目标函数（使用x[0],x[1]...格式，求最小值）",
                                  value="sum(xi**2 for xi in x)",
                                  help="示例：sum((x[i]-i)**2 for i in range(len(x)))")

        # 边界设置
        bounds = []
        for i in range(st.session_state.get("heuristic_dim", 2)):
            col1, col2 = st.columns(2)
            with col1:
                lb = st.number_input(f"x[{i}]下界", -100.0, 100.0, -5.12, key=f"lb_{i}")
            with col2:
                ub = st.number_input(f"x[{i}]上界", -100.0, 100.0, 5.12, key=f"ub_{i}")
            bounds.append((lb, ub))


    elif problem_type == "TSP问题":
        num_cities = st.number_input("城市数量", 3, 50, 4, key="heuristic_tsp_cities")

        # 新的矩阵文本输入
        distance_text = st.text_area(
            "距离矩阵（需为对称矩阵，每行用空格分隔，INF表示无路径）",
            height=200,
            value="0 INF 5 8\nINF 0 3 7\n5 3 0 INF\n8 7 INF 0",
            help="示例（含不可达路径）：\n0 INF 5 8\nINF 0 3 7\n5 3 0 INF\n8 7 INF 0"
        )

        # 解析文本为矩阵
        if distance_text:
            try:
                lines = [line.strip() for line in distance_text.split('\n') if line.strip()]
                num_lines = len(lines)
                if num_lines != num_cities:
                    st.error(f"输入行数({num_lines})与城市数量({num_cities})不匹配！")
                    st.stop()

                distance_matrix = np.full((num_cities, num_cities), 1e10)  # 默认不可达
                for i, line in enumerate(lines):
                    parts = line.split()
                    for j, val in enumerate(parts[:num_cities]):
                        if val.upper() == 'INF':
                            distance_matrix[i][j] = 1e10  # 表示极大距离
                        else:
                            distance_matrix[i][j] = float(val)

                # 确保矩阵对称性和对角线为0
                np.fill_diagonal(distance_matrix, 0)
                distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
                st.session_state.heuristic_tsp_matrix = distance_matrix

            except Exception as e:
                st.error(f"矩阵解析错误: {str(e)}")
                st.stop()

# ================= 算法执行 =================
if st.button("开始优化", type="primary"):
    # 验证算法与问题类型匹配
    if (problem_type == "TSP问题" and heuristic_method != "蚁群算法(TSP)") or \
            (problem_type == "函数优化" and heuristic_method == "蚁群算法(TSP)"):
        st.error("算法与问题类型不匹配！")
        st.stop()
    try:
        # 构造目标函数
        if problem_type == "函数优化":
            objective = eval(f"lambda x: {func_input}")

        # 执行算法
        if heuristic_method == "遗传算法":
            ga = GA(func=objective,
                    n_dim=st.session_state.heuristic_dim,
                    size_pop=st.session_state.heuristic_ga_pop,
                    max_iter=st.session_state.heuristic_max_iter,
                    lb=[b[0] for b in bounds],
                    ub=[b[1] for b in bounds])
            best_x, best_y = ga.run()
            history = ga.generation_best_Y

        elif heuristic_method == "差分进化算法":
            de = DE(func=objective,
                    n_dim=st.session_state.heuristic_dim,
                    size_pop=st.session_state.heuristic_de_pop,
                    max_iter=st.session_state.heuristic_max_iter,
                    lb=[b[0] for b in bounds],
                    ub=[b[1] for b in bounds])
            best_x, best_y = de.run()
            history = de.generation_best_Y

        elif heuristic_method == "粒子群优化":
            pso = PSO(func=objective,
                      n_dim=st.session_state.heuristic_dim,
                      pop=st.session_state.heuristic_pso_pop,
                      max_iter=st.session_state.heuristic_max_iter,
                      lb=[b[0] for b in bounds],
                      ub=[b[1] for b in bounds],
                      w=st.session_state.heuristic_pso_w)
            best_x, best_y = pso.run()
            history = pso.gbest_y_hist

        elif heuristic_method == "模拟退火算法":
            sa = SA(func=objective,
                    x0=[(b[0] + b[1]) / 2 for b in bounds],
                    T_max=st.session_state.heuristic_sa_Tmax,
                    T_min=1e-7,
                    L=st.session_state.heuristic_sa_L,
                    lb=[b[0] for b in bounds],
                    ub=[b[1] for b in bounds])
            best_x, best_y = sa.run()
            history = sa.generation_best_Y


        elif heuristic_method == "蚁群算法(TSP)":
            best_path, best_length = ant_colony_optimization(
                st.session_state.heuristic_tsp_matrix,
                n_ants=st.session_state.heuristic_aco_ants,
                n_iterations=st.session_state.heuristic_max_iter,
                alpha=st.session_state.heuristic_aco_alpha,
                beta=st.session_state.heuristic_aco_beta,
                evaporation_rate=st.session_state.heuristic_aco_evap,
                Q=st.session_state.heuristic_aco_Q
            )
            best_x, best_y = best_path, best_length

        # 存储结果
        st.session_state.heuristic_result = {
            "best_x": best_x,
            "best_y": best_y,
            "history": history if heuristic_method != "蚁群算法(TSP)" else []  # 蚁群算法无历史
        }
        st.success("优化完成！")

    except Exception as e:
        st.error(f"优化失败: {str(e)}")

# ================= 结果展示 =================
if "heuristic_result" in st.session_state:
    result = st.session_state.heuristic_result
    st.subheader("优化结果")

    col1, col2 = st.columns(2)
    with col1:
        if heuristic_method == "蚁群算法(TSP)":
            formatted_path = [int(city) for city in result['best_x']]
            readable_path = " → ".join([f"城市{city}" for city in formatted_path])
            readable_path += f" → 城市{formatted_path[0]}"  # 闭环路径标识
            st.write(f"最优路径顺序：{readable_path}")
            st.write(f"路径总长度：{result['best_y']:.2f}")
        else:
            st.metric("最优值", f"{result['best_y']}")
            st.write("最优解：", result['best_x'])

    with col2:
        if heuristic_method != "蚁群算法(TSP)":
            st.line_chart(pd.DataFrame(result['history'], columns=["适应度"]))

        if heuristic_method == "蚁群算法(TSP)":
            # 显示路径可视化
            if 'heuristic_tsp_points' in st.session_state:
                points = st.session_state.heuristic_tsp_points
                path = result['best_x']
                df = pd.DataFrame(points, columns=['x', 'y'])
                df['order'] = path + [path[0]]  # 闭环路径
                st.line_chart(df.set_index('order')[['x', 'y']])

        # 修改后的Excel导出
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({
                "变量": [f"x{i}" for i in range(len(result['best_x']))],
                "值": result['best_x']
            }).to_excel(writer, sheet_name='优化结果', index=False)

            if heuristic_method != "蚁群算法(TSP)":
                pd.DataFrame(result['history'], columns=["适应度"]).to_excel(
                    writer, sheet_name='迭代历史', index=False
                )

        output.seek(0)

        st.download_button(
            label="下载结果（Excel）",
            data=output,
            file_name=f"{heuristic_method}_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ================= 示例说明 =================
with st.expander("示例配置指南", expanded=False):
    st.markdown("""
    **函数优化示例**：
    ```python
    # Sphere函数
    sum(xi**2 for xi in x)
    # 边界设置
    x[0] ∈ [-5.12, 5.12], x[1] ∈ [-5.12, 5.12]
    ```

    **TSP问题示例**：
    ```python
    # 距离矩阵示例（4个城市）
    0 INF 5 8    # 1号城市无法直达2号
    INF 0 3 7    # 2号城市无法直达1号
    5 3 0 INF    # 3号城市无法直达4号
    8 7 INF 0    # 4号城市无法直达3号
    ```
    """)