import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_data(file_path='student_data.csv'):
    """加载数据文件"""
    try:
        data = pd.read_csv(file_path)
        print(f"数据加载成功，共 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None

def preprocess_data(data):
    """数据预处理"""
    # 处理缺失值
    data = data.dropna()
    
    # 转换性别为数值
    data['gender_numeric'] = data['gender'].map({'M': 1, 'F': 0})
    
    # 删除性别转换后可能产生的缺失行
    data = data.dropna(subset=['gender_numeric'])
    
    return data

def analyze_feature_significance(data, feature_name='height', bins=3):
    """
    使用列联表和卡方检验分析特征对预测准确性的影响
    
    参数:
    data - 数据集DataFrame
    feature_name - 要分析的特征名称 (例如 'height')
    bins - 将连续变量分箱的数量，用于创建列联表
    
    返回:
    chi2 - 卡方统计量
    p - p值
    contingency_table - 列联表
    """
    # 预处理数据
    data = preprocess_data(data)
    
    # 分割特征和目标
    X = data[['height', 'gender_numeric']]
    y = data['weight']
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建和训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测权重
    y_pred = model.predict(X_test)
    
    # 计算每个样本的预测误差
    errors = np.abs(y_test - y_pred)
    
    # 将测试集与误差结合
    test_results = X_test.copy()
    test_results['actual_weight'] = y_test.values
    test_results['predicted_weight'] = y_pred
    test_results['error'] = errors
    
    # 计算误差的中位数，用于将误差分类为高/低
    median_error = np.median(errors)
    test_results['high_error'] = test_results['error'] > median_error
    
    # 将特征值分箱
    if feature_name in ['height', 'weight', 'actual_weight']:
        # 对连续变量进行分箱
        feature_bins = pd.qcut(test_results[feature_name], bins, labels=False)
        test_results[f'{feature_name}_bin'] = feature_bins
    else:
        # 对离散变量直接使用原值
        test_results[f'{feature_name}_bin'] = test_results[feature_name]
    
    # 创建列联表：特征箱 vs 误差高低
    contingency_table = pd.crosstab(
        test_results[f'{feature_name}_bin'], 
        test_results['high_error'],
        rownames=[f'{feature_name} 分箱'],
        colnames=['高误差']
    )
    
    print(f"\n{feature_name} 与预测误差的列联表:")
    print(contingency_table)
    
    # 进行卡方独立性检验
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\n卡方检验结果:")
    print(f"卡方值: {chi2:.2f}")
    print(f"自由度: {dof}")
    print(f"p值: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print(f"\n结论: 有足够证据表明 {feature_name} 与预测误差存在显著关联 (p < {alpha})。")
        print(f"这表明 {feature_name} 对预测准确性有显著影响。")
    else:
        print(f"\n结论: 没有足够证据表明 {feature_name} 与预测误差存在显著关联 (p >= {alpha})。")
        print(f"这表明 {feature_name} 对预测准确性可能没有显著影响。")
        
    return chi2, p, contingency_table

def compare_models_with_without_feature(data, feature_to_test='gender_numeric'):
    """
    比较包含和不包含特定特征的模型性能差异
    
    参数:
    data - 数据集DataFrame
    feature_to_test - 要测试的特征名称
    
    返回:
    mae_with - 包含特征的模型MAE
    mae_without - 不包含特征的模型MAE
    """
    # 预处理数据
    data = preprocess_data(data)
    
    # 准备特征和目标
    if feature_to_test == 'height':
        X_with = data[['height', 'gender_numeric']]
        X_without = data[['gender_numeric']]
    else:  # feature_to_test == 'gender_numeric'
        X_with = data[['height', 'gender_numeric']]
        X_without = data[['height']]
    
    y = data['weight']
    
    # 分割训练和测试集
    X_with_train, X_with_test, X_without_train, X_without_test, y_train, y_test = train_test_split(
        X_with, X_without, y, test_size=0.2, random_state=42
    )
    
    # 构建和训练包含特征的模型
    model_with = LinearRegression()
    model_with.fit(X_with_train, y_train)
    y_pred_with = model_with.predict(X_with_test)
    mae_with = mean_absolute_error(y_test, y_pred_with)
    
    # 构建和训练不包含特征的模型
    model_without = LinearRegression()
    model_without.fit(X_without_train, y_train)
    y_pred_without = model_without.predict(X_without_test)
    mae_without = mean_absolute_error(y_test, y_pred_without)
    
    # 计算MAE差异的百分比
    mae_diff_percent = ((mae_without - mae_with) / mae_with) * 100
    
    print(f"\n包含特征 '{feature_to_test}' 的模型 MAE: {mae_with:.2f}")
    print(f"不包含特征 '{feature_to_test}' 的模型 MAE: {mae_without:.2f}")
    
    if mae_with < mae_without:
        print(f"结论: 包含特征 '{feature_to_test}' 的模型性能更好，MAE降低了 {abs(mae_diff_percent):.2f}%")
    else:
        print(f"结论: 不包含特征 '{feature_to_test}' 的模型性能更好，MAE降低了 {abs(mae_diff_percent):.2f}%")
    
    return mae_with, mae_without

def visualize_feature_impact(data, feature_name='height', bins=5):
    """可视化特征对预测准确性的影响"""
    # 预处理数据
    data = preprocess_data(data)
    
    # 分割特征和目标
    X = data[['height', 'gender_numeric']]
    y = data['weight']
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建和训练模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测权重
    y_pred = model.predict(X_test)
    
    # 计算每个样本的预测误差
    errors = np.abs(y_test - y_pred)
    
    # 将测试集与误差结合
    test_results = X_test.copy()
    test_results['actual_weight'] = y_test.values
    test_results['predicted_weight'] = y_pred
    test_results['error'] = errors
    
    # 对特征值进行分箱
    if feature_name in ['height', 'weight', 'actual_weight']:
        bin_labels, bin_edges = pd.qcut(test_results[feature_name], bins, retbins=True, labels=False)
        test_results[f'{feature_name}_bin'] = bin_labels
        
        # 计算每个箱的平均误差
        bin_errors = test_results.groupby(f'{feature_name}_bin')['error'].mean()
        
        # 准备箱标签（使用箱的中点值）
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(bin_midpoints)), bin_errors, width=0.7)
        plt.xlabel(f'{feature_name}')
        plt.ylabel('平均绝对误差')
        plt.title(f'{feature_name} 对预测误差的影响')
        plt.xticks(range(len(bin_midpoints)), [f"{mp:.1f}" for mp in bin_midpoints], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{feature_name}_impact.png')
        print(f"已保存图表到 {feature_name}_impact.png")
        
        return bin_errors
    else:
        # 对分类变量，直接分组计算平均误差
        bin_errors = test_results.groupby(feature_name)['error'].mean()
        
        # 绘制图表
        plt.figure(figsize=(8, 5))
        bin_errors.plot(kind='bar')
        plt.xlabel(f'{feature_name}')
        plt.ylabel('平均绝对误差')
        plt.title(f'{feature_name} 对预测误差的影响')
        plt.tight_layout()
        plt.savefig(f'{feature_name}_impact.png')
        print(f"已保存图表到 {feature_name}_impact.png")
        
        return bin_errors

def main():
    """主函数"""
    print("特征重要性分析 - 基于列联表和卡方检验")
    print("-" * 50)
    
    # 加载数据
    data_file = input("请输入数据文件路径 (默认 'student_data.csv'): ").strip()
    if not data_file:
        data_file = 'student_data.csv'
    
    data = load_data(data_file)
    if data is None:
        return
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(data.describe())
    
    while True:
        print("\n请选择操作:")
        print("1 - 分析身高(height)特征的重要性")
        print("2 - 分析性别(gender)特征的重要性")
        print("3 - 比较包含/不包含身高的模型性能")
        print("4 - 比较包含/不包含性别的模型性能")
        print("5 - 可视化身高对预测准确性的影响")
        print("6 - 可视化性别对预测准确性的影响")
        print("0 - 退出")
        
        choice = input("请输入选项编号: ").strip()
        
        if choice == '1':
            analyze_feature_significance(data, 'height')
        elif choice == '2':
            analyze_feature_significance(data, 'gender_numeric')
        elif choice == '3':
            compare_models_with_without_feature(data, 'height')
        elif choice == '4':
            compare_models_with_without_feature(data, 'gender_numeric')
        elif choice == '5':
            visualize_feature_impact(data, 'height')
        elif choice == '6':
            visualize_feature_impact(data, 'gender_numeric')
        elif choice == '0':
            break
        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main() 