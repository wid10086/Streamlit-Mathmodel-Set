import numpy as np
import streamlit as st
from scipy.optimize import linprog, minimize
from io import BytesIO
import pandas as pd


# ================= 核心算法函数（保持目标代码原样） =================
def linear_programming(c, A_ub, b_ub, A_eq=None, b_eq=None, bounds=None):
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return {
        'success': result.success,
        'x': result.x.tolist(),
        'fun': float(result.fun),
        'message': result.message
    }


def nonlinear_programming(objective_func, x0, constraints=None, bounds=None, method='SLSQP'):
    result = minimize(fun=objective_func, x0=x0, constraints=constraints, bounds=bounds, method=method)
    return {
        'success': result.success,
        'x': result.x.tolist(),
        'fun': float(result.fun),
        'message': result.message
    }


def validate_linear_inputs(c, A_ub, A_eq, bounds):
    """验证线性规划输入参数的一致性"""
    n_vars = len(c)

    # 验证约束矩阵维度
    if A_ub and any(len(row) != n_vars for row in A_ub):
        raise ValueError(f"不等式约束矩阵A_ub的列数（{len(A_ub[0])}）与目标函数系数数量（{n_vars}）不一致")
    if A_eq and any(len(row) != n_vars for row in A_eq):
        raise ValueError(f"等式约束矩阵A_eq的列数（{len(A_eq[0])}）与目标函数系数数量（{n_vars}）不一致")

    # 验证边界数量
    if bounds and len(bounds) != n_vars:
        raise ValueError(f"边界数量（{len(bounds)}）与变量数量（{n_vars}）不一致")

    return n_vars


def validate_nonlinear_inputs(x0, bounds):
    """验证非线性规划输入参数的一致性"""
    n_vars = len(x0)

    # 验证边界数量
    if bounds and len(bounds) != n_vars:
        raise ValueError(f"边界数量（{len(bounds)}）与变量数量（{n_vars}）不一致")

    return n_vars

# ================= 页面布局 =================
st.set_page_config(layout="wide")
st.title("📐 优化运筹")

# ================= 方法选择 =================
method = st.selectbox("🧮 选择求解方法", ["线性规划", "非线性规划"])

# ================= 示例说明 =================
with st.expander("💡 示例说明(inf为正无穷大)", expanded=False):
    if method == "线性规划":
        st.markdown("""
        **示例问题**:
        ```
        min 2x1 + 3x2 + x3
        s.t. x1 + x2 + x3 <= 30
            2x1 + 5x2 + 3x3 >= 10
            x1 + 3x2 = 8
            5 <= x1 <= 15
            0 <= x2 <= 10
            x3 >= 0
        ```
        """)
    elif method == "非线性规划":
        st.markdown("""
        **示例问题（非多项式问题公式表达形式需符合python语法）**:
        ```
        min (x1-2)^2 + (x2-3)^2 + (x3+1)^2
        s.t. x1^2 + x2^2 + x3^2 <= 16
            x1 + x2 + x3 >= 1
            x1 * x2 * x3 = 2
            -5 <= x1,x2,x3 <=5
        ```
        非多项式时
        ```
        min e^(x1/2) + x2*sin(x3) 
        s.t. x1^2 + ln(x2+1) >= 5
             x3/(x1+0.1) = 1  
             0.5 <= x1 <= 3
             0 <= x2 <= 4
             -π <= x3 <= π
        
        目标函数输入框：np.exp(x[0]/2) + x[1]*np.sin(x[2])
        约束条件输入框：ineq,x[0]**2 + np.log(x[1]+1) - 5
                     eq,x[2]/(x[0]+0.1) - 1
        变量边界输入：0.5,3
                    0,4
                    -3.1416,3.1416
        ```
        """)

# ================= 动态参数输入 =================
with st.expander("⚙️ 问题配置（需用英文输入法逗号）", expanded=True):
    if method == "线性规划":
        # 示例数据
        example_lp = {
            "c": [2, 3, 1],
            "A_ub": [[1, 1, 1], [-2, -5, -3]],
            "b_ub": [30, -10],
            "A_eq": [[1, 3, 0]],
            "b_eq": [8],
            "bounds": [(5, 15), (0, 10), (0, float('inf'))]
        }

        # 输入组件
        cols = st.columns(2)
        with cols[0]:
            c_input = st.text_area("目标函数系数 c（逗号分隔）", value=", ".join(map(str, example_lp["c"])))
            A_ub_input = st.text_area("不等式约束矩阵 A_ub（每行逗号分隔）",
                                      value="\n".join([", ".join(map(str, row)) for row in example_lp["A_ub"]]))
            b_ub_input = st.text_area("不等式约束向量 b_ub（逗号分隔）", value=", ".join(map(str, example_lp["b_ub"])))
        with cols[1]:
            A_eq_input = st.text_area("等式约束矩阵 A_eq（每行逗号分隔）",
                                      value="\n".join([", ".join(map(str, row)) for row in example_lp["A_eq"]]))
            b_eq_input = st.text_area("等式约束向量 b_eq（逗号分隔）", value=", ".join(map(str, example_lp["b_eq"])))
            bounds_input = st.text_area("变量边界 bounds（每行格式：min,max）",
                                        value="\n".join([f"{a},{b}" for a, b in example_lp["bounds"]]))


        # 数据转换函数
        def parse_input(text, matrix=False):
            try:
                if matrix:
                    return [list(map(float, row.split(","))) for row in text.strip().split("\n") if row]
                return list(map(float, text.split(",")))
            except:
                return None

    elif method == "非线性规划":
        # 示例数据
        example_nlp = {
            "x0": [1, 1, 2],
            "bounds": [(-5, 5), (-5, 5), (-5, 5)],
            "constraints": [
                {"type": "ineq", "func": "16 - (x[0]**2 + x[1]**2 + x[2]**2)"},
                {"type": "ineq", "func": "x[0] + x[1] + x[2] - 1"},
                {"type": "eq", "func": "x[0] * x[1] * x[2] - 2"}
            ]
        }

        # 输入组件
        cols = st.columns(2)
        with cols[0]:
            obj_func = st.text_area("目标函数 f(x)", value="(x[0]-2)**2 + (x[1]-3)**2 + (x[2]+1)**2")
            x0_input = st.text_area("初始猜测解 x0（逗号分隔，且满足约束。部分问题可能对初始敏感，可能为局部最优解，启发式算法可一定程度避免该问题）", value=", ".join(map(str, example_nlp["x0"])))
        with cols[1]:
            bounds_input = st.text_area("变量边界 bounds（每行格式：min,max）",
                                        value="\n".join([f"{a},{b}" for a, b in example_nlp["bounds"]]))
            constraints_input = st.text_area("约束条件（每行格式：type,func，ineq为>0，eq为=0）", value="\n".join(
                [f"{c['type']},{c['func']}" for c in example_nlp["constraints"]]))

# ================= 求解执行 =================
if st.button("🚀 运行求解"):
    try:
        if method == "线性规划":
            # 解析输入
            c = parse_input(c_input)
            A_ub = parse_input(A_ub_input, matrix=True)
            b_ub = parse_input(b_ub_input)
            A_eq = parse_input(A_eq_input, matrix=True) if A_eq_input else None
            b_eq = parse_input(b_eq_input) if b_eq_input else None
            bounds = [tuple(map(float, row.split(","))) for row in bounds_input.split("\n")] if bounds_input else None

            # 执行求解
            result = linear_programming(c, A_ub, b_ub, A_eq, b_eq, bounds)
            # 新增验证步骤
            n_vars = validate_linear_inputs(c, A_ub, A_eq, bounds)
            st.session_state["optimize_n_vars"] = n_vars  # 存储变量数量

        elif method == "非线性规划":
            # 动态创建函数
            objective = eval(f"lambda x: {obj_func}")
            constraints = []
            for line in constraints_input.split("\n"):
                if line:
                    constr_type, func = line.split(",", 1)
                    constraints.append({
                        "type": constr_type.strip(),
                        "fun": eval(f"lambda x: {func.strip()}")
                    })
            x0 = list(map(float, x0_input.split(",")))
            bounds = [tuple(map(float, row.split(","))) for row in bounds_input.split("\n")] if bounds_input else None

            # 执行求解
            result = nonlinear_programming(objective, x0, constraints, bounds)
            # 新增验证步骤
            n_vars = validate_nonlinear_inputs(x0, bounds)
            st.session_state["optimize_n_vars"] = n_vars  # 存储变量数量

        # 存储结果到会话状态
        st.session_state[f"optimize_{method}_result"] = result
        st.success("求解成功！")
    except Exception as e:
        st.error(f"求解失败: {str(e)}")
        st.stop()

# ================= 结果显示 =================
if f"optimize_{method}_result" in st.session_state:
    result = st.session_state[f"optimize_{method}_result"]
    n_vars = st.session_state.get("optimize_n_vars", 0)

    with st.expander("📈 求解结果", expanded=True):
        st.write(f"**变量数量**: {n_vars}")
        st.write(f"**求解状态**: {'成功' if result['success'] else '失败'}")
        st.write(f"**最优解**: {result['x']}")
        st.write(f"**最优值**: {result['fun']:.4f}")
        st.write(f"**状态信息**: {result['message']}")

        # 导出Excel
        df = pd.DataFrame({
            "变量": [f"x{i + 1}" for i in range(len(result['x']))],
            "最优值": result['x']
        })
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        st.download_button(
            label="⬇️ 下载结果",
            data=excel_buffer.getvalue(),
            file_name=f"{method}_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
