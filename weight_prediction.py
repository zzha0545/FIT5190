import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # 暂时不需要绘图功能
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def load_data(file_path):
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 不存在")
    
    data = pd.read_csv(file_path)
    print(f"数据加载成功，共 {len(data)} 条记录")
    return data

def preprocess_data(data):
    """数据预处理"""
    # 检查并打印缺失值情况
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print("数据中存在缺失值:")
        print(missing_values)
        print("进行处理...")
        
        # 删除包含缺失值的行
        data = data.dropna()
        print(f"处理后数据量: {len(data)}")
    else:
        print("数据中没有缺失值")
    
    # 确保列中的数据类型正确
    data['height'] = pd.to_numeric(data['height'], errors='coerce')
    data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
    
    # 处理转换后可能产生的新缺失值
    if data.isnull().sum().sum() > 0:
        print("数据类型转换后出现缺失值，再次处理...")
        data = data.dropna()
        print(f"处理后数据量: {len(data)}")
    
    # 性别特征转换为数值
    data['gender_numeric'] = data['gender'].map({'M': 1, 'F': 0})
    
    # 检查性别映射后是否有缺失值（可能由于CSV中有非M/F的值导致）
    if data['gender_numeric'].isnull().sum() > 0:
        print("性别数据中有非法值，进行处理...")
        # 打印出有问题的性别值
        print("非法性别值:")
        print(data.loc[data['gender_numeric'].isnull(), 'gender'].unique())
        # 删除这些行
        data = data.dropna(subset=['gender_numeric'])
        print(f"处理后数据量: {len(data)}")
    
    return data

def train_model(data):
    """训练线性回归模型"""
    # 准备特征和目标变量
    X = data[['height', 'gender_numeric']]  # 特征：身高和性别
    y = data['weight']  # 目标：体重
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"模型评估:")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"决定系数 (R²): {r2:.2f}")
    
    # 提取系数和截距
    # W = [w_height, w_gender]
    # ε = bias
    w_height = model.coef_[0]
    w_gender = model.coef_[1]
    bias = model.intercept_
    
    print(f"线性模型: 体重 = {w_height:.2f} × 身高 + {w_gender:.2f} × 性别 + {bias:.2f}")
    print(f"其中性别编码: 男性(M)=1, 女性(F)=0")
    
    return model, X_test, y_test, y_pred

def visualize_results(data, model, X_test, y_test, y_pred):
    """可视化结果（已注释掉图像生成部分）"""
    # 计算并显示一些基本的统计信息代替图表
    residuals = y_test - y_pred
    print(f"测试集样本数: {len(y_test)}")
    print(f"预测残差平均值: {np.mean(residuals):.2f}")
    print(f"预测残差标准差: {np.std(residuals):.2f}")
    print(f"预测平均绝对误差: {np.mean(np.abs(residuals)):.2f} kg")
    print(f"预测最大绝对误差: {np.max(np.abs(residuals)):.2f} kg")
    
    # 计算按性别分组的预测精度
    gender_col = []
    for idx in X_test.index:
        gender_col.append(data.loc[idx, 'gender'])
    
    male_residuals = residuals[np.array(gender_col) == 'M']
    female_residuals = residuals[np.array(gender_col) == 'F']
    
    if len(male_residuals) > 0:
        print(f"男性样本预测平均绝对误差: {np.mean(np.abs(male_residuals)):.2f} kg")
    if len(female_residuals) > 0:
        print(f"女性样本预测平均绝对误差: {np.mean(np.abs(female_residuals)):.2f} kg")
    
    # 以下是原可视化代码（已注释）
    """
    plt.figure(figsize=(12, 10))
    
    # 散点图：身高与体重的关系，按性别区分
    plt.subplot(2, 2, 1)
    for gender, color, label in zip(['M', 'F'], ['blue', 'red'], ['男性', '女性']):
        mask = data['gender'] == gender
        plt.scatter(data.loc[mask, 'height'], data.loc[mask, 'weight'], 
                    color=color, label=label, alpha=0.6)
    
    plt.xlabel('身高 (cm)')
    plt.ylabel('体重 (kg)')
    plt.title('身高与体重关系（按性别区分）')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 预测值与实际值对比
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('实际体重 (kg)')
    plt.ylabel('预测体重 (kg)')
    plt.title('预测值与实际值对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 残差图
    plt.subplot(2, 2, 3)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('预测体重 (kg)')
    plt.ylabel('残差')
    plt.title('残差分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 残差分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差直方图')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('weight_prediction_results.png')
    plt.show()
    print("结果图表已保存为 'weight_prediction_results.png'")
    """

def make_prediction(model, height, gender):
    """使用模型进行预测"""
    gender_numeric = 1 if gender.upper() == 'M' else 0
    prediction = model.predict([[height, gender_numeric]])[0]
    return prediction

def batch_test(model, test_file):
    """批量测试函数：读取测试数据CSV文件并计算MAE
    
    参数:
    model - 训练好的线性回归模型
    test_file - 测试数据CSV文件路径
    
    返回:
    mae - 平均绝对误差
    predictions - 包含原始数据和预测结果的DataFrame
    """
    try:
        # 读取测试数据
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试数据文件 {test_file} 不存在")
        
        test_data = pd.read_csv(test_file)
        print(f"测试数据加载成功，共 {len(test_data)} 条记录")
        
        # 数据预处理
        test_data = preprocess_data(test_data)
        
        if len(test_data) == 0:
            raise ValueError("预处理后没有可用的测试数据")
            
        # 准备特征
        X_test = test_data[['height', 'gender_numeric']]
        
        # 如果测试数据中包含体重列，可以计算预测准确度
        if 'weight' in test_data.columns:
            y_true = test_data['weight']
            
            # 进行预测
            y_pred = model.predict(X_test)
            
            # 计算MAE
            mae = mean_absolute_error(y_true, y_pred)
            print(f"测试集平均绝对误差(MAE): {mae:.2f} kg")
            
            # 将预测结果添加到原始数据中
            test_data['predicted_weight'] = y_pred
            test_data['error'] = test_data['weight'] - test_data['predicted_weight']
            test_data['abs_error'] = abs(test_data['error'])
            
            # 显示误差统计
            print(f"最大绝对误差: {test_data['abs_error'].max():.2f} kg")
            print(f"最小绝对误差: {test_data['abs_error'].min():.2f} kg")
            print(f"误差标准差: {test_data['error'].std():.2f} kg")
            
            # 按性别分组计算MAE
            gender_mae = test_data.groupby('gender')['abs_error'].mean()
            for gender, mae_value in gender_mae.items():
                print(f"{gender}性平均绝对误差: {mae_value:.2f} kg")
                
            return mae, test_data
        else:
            # 如果没有体重列，只进行预测
            y_pred = model.predict(X_test)
            test_data['predicted_weight'] = y_pred
            print("测试数据中没有体重列，只生成预测结果")
            return None, test_data
            
    except Exception as e:
        print(f"批量测试出错: {e}")
        return None, None

def main():
    """主函数"""
    print("体重预测线性模型 - 基于身高和性别")
    print("-" * 50)
    
    # 加载数据
    try:
        data_file = input("请输入训练数据文件路径 (默认 'student_data.csv'): ").strip()
        if not data_file:
            data_file = 'student_data.csv'
        
        data = load_data(data_file)
        
        # 显示数据基本信息
        print("\n数据预览:")
        print(data.head())
        print("\n数据统计描述:")
        print(data.describe())
        
        # 数据预处理
        data = preprocess_data(data)
        
        # 训练模型
        print("\n训练模型...")
        model, X_test, y_test, y_pred = train_model(data)
        
        # 显示模型评估指标（不生成图表）
        print("\n模型评估指标:")
        visualize_results(data, model, X_test, y_test, y_pred)
        
        # 功能选择
        while True:
            print("\n请选择操作:")
            print("1. 单个预测（输入身高和性别）")
            print("2. 批量测试（从CSV文件）")
            print("3. 退出程序")
            
            choice = input("请输入选项编号 (1-3): ").strip()
            
            if choice == "1":
                # 交互式预测
                print("\n进行单个预测:")
                try:
                    height = float(input("请输入身高(cm): "))
                    gender = input("请输入性别(M/F): ").strip().upper()
                    
                    if gender not in ['M', 'F']:
                        print("性别输入错误，请输入 M 或 F")
                        continue
                    
                    predicted_weight = make_prediction(model, height, gender)
                    print(f"预测体重: {predicted_weight:.2f} kg")
                
                except ValueError:
                    print("输入无效，请重试")
                
            elif choice == "2":
                # 批量测试
                test_file = input("请输入测试数据CSV文件路径: ").strip()
                if not test_file:
                    print("文件路径不能为空")
                    continue
                
                mae, predictions = batch_test(model, test_file)
                
                if predictions is not None:
                    # 显示预测结果
                    print("\n预测结果预览 (前5条):")
                    print(predictions.head())
                    
                    # 询问是否保存结果
                    save_option = input("\n是否保存预测结果? (y/n): ").strip().lower()
                    if save_option == 'y':
                        output_file = input("请输入输出文件名 (默认 'prediction_results.csv'): ").strip()
                        if not output_file:
                            output_file = 'prediction_results.csv'
                        
                        predictions.to_csv(output_file, index=False)
                        print(f"预测结果已保存到 {output_file}")
            
            elif choice == "3":
                print("退出程序")
                break
            
            else:
                print("无效选项，请重新输入")
    
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"程序出错: {e}")

if __name__ == "__main__":
    main() 