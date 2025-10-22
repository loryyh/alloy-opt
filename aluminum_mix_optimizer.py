import numpy as np
import pandas as pd
from scipy.optimize import linprog
import streamlit as st

class AlloyOptimizer:
    def __init__(self):
        # 预设15种原材料类型及其默认参数
        self.material_presets = {
            '易拉罐': {
                'Al': 97.5, 'Cu': 0.1, 'Mg': 1.0, 'Zn': 1.0,
                'price': 15.0, 'density': 2.7
            },
            '线缆铝芯': {
                'Al': 99.0, 'Cu': 0.3, 'Mg': 0.3, 'Zn': 0.2,
                'price': 18.0, 'density': 2.7
            },
            '建筑铝型材': {
                'Al': 95.0, 'Cu': 0.5, 'Mg': 1.2, 'Zn': 3.0,
                'price': 16.5, 'density': 2.7
            },
            '航空级铝废料': {
                'Al': 90.0, 'Cu': 2.0, 'Mg': 5.0, 'Zn': 2.5,
                'price': 25.0, 'density': 2.8
            },
            '汽车铝板': {
                'Al': 93.0, 'Cu': 0.8, 'Mg': 1.5, 'Zn': 4.5,
                'price': 17.0, 'density': 2.7
            },
            '铝合金轮毂': {
                'Al': 92.0, 'Cu': 0.5, 'Mg': 2.0, 'Zn': 5.0,
                'price': 19.0, 'density': 2.8
            },
            '电子外壳': {
                'Al': 96.0, 'Cu': 0.2, 'Mg': 0.5, 'Zn': 3.0,
                'price': 16.0, 'density': 2.7
            },
            '厨具铝材': {
                'Al': 98.0, 'Cu': 0.1, 'Mg': 0.4, 'Zn': 1.2,
                'price': 14.5, 'density': 2.7
            },
            '船舶铝材': {
                'Al': 94.0, 'Cu': 0.3, 'Mg': 3.0, 'Zn': 2.5,
                'price': 20.0, 'density': 2.7
            },
            '高铁铝材': {
                'Al': 91.0, 'Cu': 1.0, 'Mg': 4.0, 'Zn': 3.5,
                'price': 22.0, 'density': 2.8
            },
            '包装铝箔': {
                'Al': 99.5, 'Cu': 0.05, 'Mg': 0.1, 'Zn': 0.3,
                'price': 13.0, 'density': 2.7
            },
            '散热器铝材': {
                'Al': 97.0, 'Cu': 0.8, 'Mg': 0.5, 'Zn': 1.5,
                'price': 15.5, 'density': 2.7
            },
            '军工铝材': {
                'Al': 89.0, 'Cu': 2.5, 'Mg': 6.0, 'Zn': 2.0,
                'price': 28.0, 'density': 2.8
            },
            '建筑模板': {
                'Al': 95.5, 'Cu': 0.4, 'Mg': 1.0, 'Zn': 2.8,
                'price': 16.0, 'density': 2.7
            },
            '通用铝合金': {
                'Al': 93.5, 'Cu': 0.6, 'Mg': 1.8, 'Zn': 3.8,
                'price': 17.5, 'density': 2.7
            }
        }
        
        self.elements = ['Al', 'Cu', 'Mg', 'Zn']
    
    def optimize_alloy(self, materials_data, order_requirements, max_ratio_constraint=True):
        n_materials = len(materials_data)
        
        # 目标函数系数 (成本最小化)
        c = [material['price'] for material in materials_data]
        
        # 约束条件
        A_eq = []  # 等式约束
        b_eq = []  # 等式约束右侧
        
        A_ub = []  # 不等式约束
        b_ub = []  # 不等式约束右侧
        
        total_w = order_requirements['total_weight']      # 先缓存总量
        # 1. 总量约束: 所有原材料重量之和 = 订单重量
        A_eq.append([1] * n_materials)
        b_eq.append(order_requirements['total_weight'])
        
        # 2. 元素含量约束
        for element in self.elements:
            # 元素总含量约束 (允许±0.5%的误差)
            target_min = (order_requirements['element_content'][element] - 0.5) / 100   # %
            target_max = (order_requirements['element_content'][element] + 0.5) / 100
            coeffs = [material['element_content'][element]/100 for material in materials_data]

            A_ub.append([-c for c in coeffs])
            b_ub.append(-target_min * total_w)
            A_ub.append(coeffs)
            b_ub.append(target_max * total_w)
        
        # 3. 库存约束
        for i, material in enumerate(materials_data):
            A_ub.append([0]*i + [1] + [0]*(n_materials-1-i))
            b_ub.append(material['stock'])
        
        # 4. 最大比例约束 (如果原材料供货不足)
        if max_ratio_constraint:
            # 找出库存最少的两种材料
            stocks = [(i, material['stock']) for i, material in enumerate(materials_data)]
            stocks.sort(key=lambda x: x[1])
            
            if len(stocks) >= 2:
                # 对库存最少的两种材料设置最大比例约束 (不超过30%)
                for idx, _ in stocks[:2]:
                    constraint = [0] * n_materials
                    constraint[idx] = 1
                    A_ub.append(constraint)
                    b_ub.append(order_requirements['total_weight'] * 0.3)
        
        # 变量边界 (非负)
        bounds = [(0, None) for _ in range(n_materials)]
        
        # 求解线性规划
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                return {
                    'success': True,
                    'total_cost': result.fun,
                    'material_weights': result.x,
                    'message': '优化成功'
                }
            else:
                return {
                    'success': False,
                    'message': f'无法找到可行解: {result.message}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'求解过程中出错: {str(e)}'
            }
    
    def calculate_final_composition(self, materials_data, weights):
        """计算最终合金成分"""
        total_weight = sum(weights)
        if total_weight == 0:
            return None
            
        final_comp = {element: 0 for element in self.elements}
        
        for i, material in enumerate(materials_data):
            for element in self.elements:
                element_weight = weights[i] * material['element_content'][element] / 100
                final_comp[element] += element_weight
        
        # 转换为百分比
        for element in final_comp:
            final_comp[element] = (final_comp[element] / total_weight) * 100
            
        return final_comp

def initialize_session_state():
    """初始化session state"""
    if 'materials_list' not in st.session_state:
        st.session_state.materials_list = []
    if 'next_material_id' not in st.session_state:
        st.session_state.next_material_id = 1
    # 不再固定默认价格，而是在选择时动态设置

def is_material_already_added(material_type):
    """检查原材料类型是否已经添加过"""
    return any(material['name'] == material_type for material in st.session_state.materials_list)

def add_material(material_type, stock_kg, custom_price, preset_data):
    """添加原材料到列表"""
    
    new_material = {
        'id': st.session_state.next_material_id,
        'name': material_type,
        'element_content': {
            'Al': preset_data['Al'],
            'Cu': preset_data['Cu'],
            'Mg': preset_data['Mg'],
            'Zn': preset_data['Zn']
        },
        'price': custom_price,
        'stock': stock_kg
    }
    
    st.session_state.materials_list.append(new_material)
    st.session_state.next_material_id += 1
    st.success(f"✅ 已成功添加 {material_type}")
    return True

def remove_material(material_id):
    """从列表中移除原材料"""
    material_to_remove = next((mat for mat in st.session_state.materials_list if mat['id'] == material_id), None)
    if material_to_remove:
        st.session_state.materials_list = [
            material for material in st.session_state.materials_list 
            if material['id'] != material_id
        ]
        st.success(f"✅ 已移除 {material_to_remove['name']}")
    else:
        st.error("❌ 未找到要移除的材料")

def get_available_material_types(optimizer):
    """获取可用的原材料类型（排除已添加的）"""
    added_types = [mat['name'] for mat in st.session_state.materials_list]
    available_types = [mat_type for mat_type in optimizer.material_presets.keys() if mat_type not in added_types]
    return available_types

def validate_order_inputs(weight, al_content, cu_content, mg_content, zn_content):
    """验证订单输入数据的有效性"""
    errors = []
    
    if weight <= 0:
        errors.append("订单重量必须大于0")
    
    if not (85.0 <= al_content <= 99.9):
        errors.append("铝含量必须在85.0%到99.9%之间")
    
    if not (0.0 <= cu_content <= 10.0):
        errors.append("铜含量必须在0.0%到10.0%之间")
    
    if not (0.0 <= mg_content <= 10.0):
        errors.append("镁含量必须在0.0%到10.0%之间")
    
    if not (0.0 <= zn_content <= 10.0):
        errors.append("锌含量必须在0.0%到10.0%之间")
    
    # 检查总含量合理性
    total_content = al_content + cu_content + mg_content + zn_content
    if total_content > 100.0:
        errors.append(f"元素总含量({total_content:.1f}%)不能超过100%")
    
    return errors

def main():
    st.set_page_config(
        page_title="铝合金原材料配比优化系统",
        # page_icon="🏭",
        layout="wide"
    )
    
    st.title("铝合金原材料配比优化系统")
    st.markdown("---")
    
    # 初始化session state
    initialize_session_state()
    
    optimizer = AlloyOptimizer()
    
    # 侧边栏 - 订单要求
    st.sidebar.header("订单要求")
    
    # 订单重量输入
    order_weight = st.sidebar.number_input(
        "订单重量 (公斤)", 
        min_value=100.0, 
        max_value=100000.0, 
        value=1000.0,
        step=100.0,
        help="请输入订单需要的总重量，范围：100-100,000公斤"
    )
    
    st.sidebar.subheader("元素含量要求 (%)")
    
    # 元素含量文本框输入
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        al_content = st.number_input(
            "铝(Al)含量 (%)",
            min_value=85.0,
            max_value=99.9,
            value=95.0,
            step=0.1,
            format="%.1f",
            help="铝含量范围：85.0%-99.9%"
        )
        
        cu_content = st.number_input(
            "铜(Cu)含量 (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            format="%.1f",
            help="铜含量范围：0.0%-10.0%"
        )
    
    with col2:
        mg_content = st.number_input(
            "镁(Mg)含量 (%)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="镁含量范围：0.0%-10.0%"
        )
        
        zn_content = st.number_input(
            "锌(Zn)含量 (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="锌含量范围：0.0%-10.0%"
        )
    
    # 显示当前设置的元素总含量
    total_content = al_content + cu_content + mg_content + zn_content
    st.sidebar.info(f"**当前元素总含量: {total_content:.1f}%**")
    
    if total_content > 100.0:
        st.sidebar.error("⚠️ 元素总含量超过100%，请调整各元素含量")
    
    # 验证输入数据
    validation_errors = validate_order_inputs(order_weight, al_content, cu_content, mg_content, zn_content)
    
    order_requirements = {
        'total_weight': order_weight,
        'element_content': {
            'Al': al_content,
            'Cu': cu_content,
            'Mg': mg_content,
            'Zn': zn_content
        }
    }
    
    # 主界面 - 原材料管理
    st.header("原材料管理")
    
    # 添加新材料表单
    st.subheader("添加新原材料")
    
    # 获取可用的原材料类型
    available_types = get_available_material_types(optimizer)
    
    if not available_types:
        st.success("所有原材料类型都已添加完毕！")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            def on_material_change():
                selected = st.session_state.material_select
                st.session_state.price_input = float(optimizer.material_presets[selected]['price'])

            material_type = st.selectbox(
                "选择原材料类型",
                available_types,
                key="material_select",
                on_change=on_material_change,
            )
        
        with col2:
            stock_kg = st.number_input("库存 (公斤)", min_value=0.0, value=1000.0, step=100.0, key="stock_input")
        
        with col3:
            # 获取当前选择材料的默认单价
            preset_data = optimizer.material_presets[material_type]
            default_price = preset_data['price']
            
            # 显示单价输入框，使用当前价格
            custom_price = st.number_input(
                "单价 (元/公斤)", 
                min_value=0.0, 
                value=float(default_price),
                step=0.5, 
                key=f"price_input_{material_type}",
                help=f"默认单价: ¥{default_price}/kg"
            )

        st.info(f"""
        **{material_type} 详细信息:**
        - 元素含量: Al: {preset_data['Al']}%, Cu: {preset_data['Cu']}%, Mg: {preset_data['Mg']}%, Zn: {preset_data['Zn']}%
        - 默认单价: ¥{preset_data['price']}/kg
        - 密度: {preset_data['density']} g/cm³
        """)
        
        # 添加材料按钮
        if st.button("添加原材料", type="primary"):
            if add_material(material_type, stock_kg, custom_price, preset_data):
                st.rerun()
    
    st.markdown("---")
    
    # 显示当前材料列表
    st.subheader("当前原材料列表")
    
    if not st.session_state.materials_list:
        st.warning("⚠️ 尚未添加任何原材料，请先在上方添加原材料")
    else:
        # 显示材料表格
        materials_df = pd.DataFrame([
            {
                'ID': mat['id'],
                '名称': mat['name'],
                'Al含量%': mat['element_content']['Al'],
                'Cu含量%': mat['element_content']['Cu'], 
                'Mg含量%': mat['element_content']['Mg'],
                'Zn含量%': mat['element_content']['Zn'],
                '单价(元/kg)': mat['price'],
                '库存(kg)': f"{mat['stock']:,.1f}"
            }
            for mat in st.session_state.materials_list
        ])
        
        # 显示表格
        st.dataframe(materials_df, use_container_width=True, hide_index=True)
        
        # 显示统计信息
        total_stock = sum(mat['stock'] for mat in st.session_state.materials_list)
        avg_price = sum(mat['price'] for mat in st.session_state.materials_list) / len(st.session_state.materials_list)
        total_value = sum(mat['price'] * mat['stock'] for mat in st.session_state.materials_list)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("已添加材料种类", len(st.session_state.materials_list))
        with col2:
            st.metric("总库存量", f"{total_stock:,.1f} kg")
        with col3:
            st.metric("平均单价", f"¥{avg_price:.2f}/kg")
        with col4:
            st.metric("库存总价值", f"¥{total_value:,.0f}")
        
        # 显示剩余可添加类型
        remaining_types = get_available_material_types(optimizer)
        if remaining_types:
            st.info(f"📋 还可添加的材料类型: {', '.join(remaining_types)}")
        
        # 添加移除功能
        st.subheader("管理原材料")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            material_options = {
                mat['id']: f"{mat['name']} (库存:{mat['stock']}kg)"
                for mat in st.session_state.materials_list
            }
            if material_options:
                material_to_remove = st.selectbox(
                    "选择要移除的原材料",
                    options=list(material_options.keys()),
                    format_func=lambda x: material_options[x]
                )
                
                if st.button("🗑️ 移除选中原材料", type="secondary"):
                    remove_material(material_to_remove)
                    st.rerun()
            else:
                st.info("暂无原材料可移除")
    
    st.markdown("---")
    
    # 优化计算
    st.header("配比优化计算")
    
    # 显示订单要求汇总
    st.subheader("当前订单要求")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("订单重量", f"{order_weight:,.0f} kg")
    with col2:
        st.metric("Al含量", f"{al_content}%")
    with col3:
        st.metric("Cu含量", f"{cu_content}%")
    with col4:
        st.metric("Mg含量", f"{mg_content}%")
    with col5:
        st.metric("Zn含量", f"{zn_content}%")
    
    if validation_errors:
        st.error("订单要求存在错误，请修正：")
        for error in validation_errors:
            st.error(f"  - {error}")
    
    if not st.session_state.materials_list:
        st.error(" 请先添加至少一种原材料才能进行优化计算")
    else:
        col1, col2 = st.columns([1, 1])
        if "opt_result" not in st.session_state:
            st.session_state.opt_result = None

        with col1:
            use_constraints = st.checkbox("启用最大比例约束", value=True)
            st.caption("当原材料供货不足时，限制单个材料的最大使用比例")
        
        with col2:
            if validation_errors:
                st.button("开始优化计算", type="primary", disabled=True,
                        help="请先修正订单要求中的错误")
            else:
                if st.button("开始优化计算", type="primary"):
                    with st.spinner("正在计算最优配比..."):
                        res = optimizer.optimize_alloy(
                            st.session_state.materials_list,
                            order_requirements,
                            max_ratio_constraint=use_constraints
                        )
                        # 把结果丢进 session_state，下方区域会自动更新
                        st.session_state.opt_result = res
                        if res["success"]:
                            st.success("优化计算完成！")

        # 3. 结果展示区（永远在最大比例复选框下方）
        st.markdown("---")   # 可有可无的分隔线
        if st.session_state.opt_result is not None:
            result = st.session_state.opt_result
            if result["success"]:
                # ===== 以下整块就是你原来的结果展示代码 =====
                weights = result["material_weights"]
                total_cost = result["total_cost"]
                final_comp = optimizer.calculate_final_composition(
                    st.session_state.materials_list, weights
                )

                st.subheader("最优配比结果")
                result_data = []
                total_used = 0
                for i, material in enumerate(st.session_state.materials_list):
                    if weights[i] > 0.001:
                        ratio = (weights[i] / order_weight) * 100
                        total_used += weights[i]
                        result_data.append({
                            '原材料': material['name'],
                            '使用量(kg)': round(weights[i], 2),
                            '比例(%)': round(ratio, 2),
                            '成本(元)': round(weights[i] * material['price'], 2)
                        })
                result_df = pd.DataFrame(result_data)
                st.dataframe(result_df, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总成本", f"¥{total_cost:,.2f}")
                with col2:
                    st.metric("单位成本", f"¥{total_cost/order_weight:.2f}/kg")
                with col3:
                    st.metric("材料总用量", f"{total_used:,.1f} kg")
                with col4:
                    utilization = (total_used / order_weight) * 100
                    st.metric("材料利用率", f"{utilization:.1f}%")

                st.subheader("最终合金成分分析")
                comp_data = []
                for element in optimizer.elements:
                    comp_data.append({
                        '元素': element,
                        '目标含量%': order_requirements['element_content'][element],
                        '实际含量%': round(final_comp[element], 2),
                        '偏差': round(final_comp[element] - order_requirements['element_content'][element], 2)
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True)

                st.subheader("成本分析")
                cost_breakdown = []
                for i, material in enumerate(st.session_state.materials_list):
                    if weights[i] > 0.001:
                        material_cost = weights[i] * material['price']
                        cost_breakdown.append({
                            '原材料': material['name'],
                            '成本占比%': round((material_cost / total_cost) * 100, 1),
                            '单位成本': f"¥{material['price']}/kg",
                            '使用量占比%': round((weights[i] / order_weight) * 100, 1)
                        })
                cost_df = pd.DataFrame(cost_breakdown)
                st.dataframe(cost_df, use_container_width=True)

            else:   # 优化失败
                st.error(f"❌ {result['message']}")

if __name__ == "__main__":
    main()