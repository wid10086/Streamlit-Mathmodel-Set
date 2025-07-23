import streamlit as st

# 页面配置
st.set_page_config(page_title="智能建模平台", layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
.algorithm-list {
    padding: 15px;
    background: #f8f9fa;
    border-radius: 10px;
    margin: 10px 0;
}
.algorithm-list li {
    padding: 8px 0;
    font-size: 1.1em;
}
.guide-section {
    background: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
st.markdown("# 智能建模平台")
st.markdown("#### 北京科技大学天码智能社")
# 主内容区
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    # 算法类别展示
    st.markdown("### 📚 算法体系")
    st.markdown("<span style='color: #888; font-size:0.98em;'>点击下方算法名称可直接跳转对应页面</span>", unsafe_allow_html=True)
    # 使用st.page_link实现点击跳转
    st.page_link("pages/1Normalize.py", label="数据归一化/标准化", icon="🧮")
    st.page_link("pages/2Regression.py", label="多元回归/拟合", icon="📈")
    st.page_link("pages/3Evaluate.py", label="多准则决策分析/评估", icon="📊")
    st.page_link("pages/4Time_Series.py", label="时间序列预测分析", icon="⏳")
    st.page_link("pages/5Data_Mining.py", label="数据挖掘分析", icon="🔎")
    st.page_link("pages/6Optimize.py", label="优化运筹", icon="📐")
    st.page_link("pages/7Heuristic.py", label="启发式算法", icon="🧠")
    st.page_link("pages/8Outlier.py", label="数据预处理/异常值处理", icon="🧹")

with col2:
    # 使用指南
    st.markdown("### 📖 使用指南")
    with st.container():
        st.markdown("""
        <div class="guide-section">
        <h4>📌 数据准备</h4>
        <p>• 各算法页面均提供示例数据集<br>
        • 支持CSV/Excel格式文件上传<br>
        • 未上传数据时默认使用示例数据</p>

        <h4>⚙️ 操作提示</h4>
        <p>• 计算出结果后及时保存<br>
        • 部分算法源代码有参数讲解，如仍不理解请善用AI<br>
        • 支持一键导出处理结果（Excel格式）</p>

        <h4>🚨 注意事项</h4>
        <p>• Streamlit框架采用「单次点击单次执行」机制<br>
        • 切换页面后算法状态自动重置<br>
        • 复杂算法建议先用示例数据验证流程</p>
        </div>
        """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<small>技术支持：天码智能社 | 版本：1.1 | 更新日期：2024-05</small>
""", unsafe_allow_html=True)