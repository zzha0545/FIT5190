#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征重要性分析示例脚本
演示如何使用列联表和卡方检验分析特征对预测准确性的影响
"""

from feature_significance_test import (
    load_data, 
    analyze_feature_significance,
    compare_models_with_without_feature,
    visualize_feature_impact
)

def main():
    print("体重预测模型 - 特征重要性分析示例")
    print("=" * 50)
    
    # 1. 加载数据
    print("加载数据...")
    data = load_data('student_data.csv')
    if data is None:
        return
        
    # 2. 使用列联表和卡方检验分析身高特征的重要性
    print("\n分析 1: 使用列联表和卡方检验分析身高特征的重要性")
    print("-" * 50)
    chi2_height, p_height, table_height = analyze_feature_significance(data, 'height', bins=3)
    
    # 3. 使用列联表和卡方检验分析性别特征的重要性
    print("\n分析 2: 使用列联表和卡方检验分析性别特征的重要性")
    print("-" * 50)
    chi2_gender, p_gender, table_gender = analyze_feature_significance(data, 'gender_numeric')
    
    # 4. 比较包含和不包含身高特征的模型性能
    print("\n分析 3: 比较包含和不包含身高特征的模型性能")
    print("-" * 50)
    mae_with_height, mae_without_height = compare_models_with_without_feature(data, 'height')
    
    # 5. 比较包含和不包含性别特征的模型性能
    print("\n分析 4: 比较包含和不包含性别特征的模型性能")
    print("-" * 50)
    mae_with_gender, mae_without_gender = compare_models_with_without_feature(data, 'gender_numeric')
    
    # 6. 可视化身高特征对预测准确性的影响
    print("\n分析 5: 可视化身高特征对预测准确性的影响")
    print("-" * 50)
    visualize_feature_impact(data, 'height', bins=5)
    
    # 7. 可视化性别特征对预测准确性的影响
    print("\n分析 6: 可视化性别特征对预测准确性的影响")
    print("-" * 50)
    visualize_feature_impact(data, 'gender_numeric')
    
    # 8. 综合分析结果
    print("\n综合分析结果:")
    print("=" * 50)
    
    # 根据p值比较身高和性别特征的重要性
    alpha = 0.05
    print(f"身高特征重要性 (卡方检验):")
    print(f"- 卡方值: {chi2_height:.2f}")
    print(f"- p值: {p_height:.4f}")
    print(f"- 显著性: {'显著' if p_height < alpha else '不显著'}")
    
    print(f"\n性别特征重要性 (卡方检验):")
    print(f"- 卡方值: {chi2_gender:.2f}")
    print(f"- p值: {p_gender:.4f}")
    print(f"- 显著性: {'显著' if p_gender < alpha else '不显著'}")
    
    # 根据MAE比较身高和性别特征的重要性
    print("\n基于模型性能比较的特征重要性:")
    
    height_impact = ((mae_without_height - mae_with_height) / mae_with_height) * 100
    gender_impact = ((mae_without_gender - mae_with_gender) / mae_with_gender) * 100
    
    print(f"去除身高特征导致MAE增加: {height_impact:.2f}%")
    print(f"去除性别特征导致MAE增加: {gender_impact:.2f}%")
    
    # 判断哪个特征更重要
    if height_impact > gender_impact and height_impact > 0:
        print("\n结论: 身高特征对预测准确性的影响更大")
    elif gender_impact > height_impact and gender_impact > 0:
        print("\n结论: 性别特征对预测准确性的影响更大")
    elif height_impact <= 0 and gender_impact <= 0:
        print("\n结论: 两个特征都不显著影响预测准确性，可能需要考虑其他特征")
    else:
        print("\n结论: 两个特征都对预测准确性有一定影响")
    
    print("\n分析完成！结果已保存为图表。")

if __name__ == "__main__":
    main() 