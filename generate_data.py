import pandas as pd
import numpy as np
import random

def generate_student_data(num_students=100, output_file='student_data.csv'):
    """生成随机学生数据并保存到CSV文件"""
    np.random.seed(42)  # 设置随机种子以便结果可复现
    
    # 男女性别分布
    genders = np.random.choice(['M', 'F'], size=num_students)
    
    # 根据性别生成合理的身高数据（单位：厘米）
    heights = []
    for gender in genders:
        if gender == 'M':
            # 男性身高范围通常在160-190厘米之间
            height = np.random.normal(175, 8)
        else:
            # 女性身高范围通常在150-175厘米之间
            height = np.random.normal(162, 7)
        heights.append(round(height, 1))
    
    # 根据身高和性别生成合理的体重数据（单位：千克）
    weights = []
    for height, gender in zip(heights, genders):
        if gender == 'M':
            # 男性体重，用身高的合理BMI值计算（BMI范围：18.5-25.0）
            bmi = np.random.uniform(20.0, 26.0)
        else:
            # 女性体重，用身高的合理BMI值计算（BMI范围：18.0-24.0）
            bmi = np.random.uniform(19.0, 24.0)
        
        # 体重(kg) = 身高(m)² × BMI
        weight = (height / 100) ** 2 * bmi
        weights.append(round(weight, 1))
    
    # 创建数据框
    data = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'gender': genders
    })
    
    # 可选：添加少量缺失值和错误值进行测试
    if num_students >= 50:
        # 随机选择几行设置缺失值
        for _ in range(3):
            row = np.random.randint(0, num_students)
            col = np.random.choice(['height', 'weight'])
            data.loc[row, col] = np.nan
        
        # 随机将1-2个性别值设为非法值（既不是M也不是F）
        for _ in range(2):
            row = np.random.randint(0, num_students)
            data.loc[row, 'gender'] = np.random.choice(['X', 'O', ''])
    
    # 保存到CSV文件
    data.to_csv(output_file, index=False)
    print(f"已生成{num_students}条学生数据并保存到 {output_file}")
    print(f"数据统计信息:")
    print(data.describe())
    print("\n男女比例:")
    print(data['gender'].value_counts())
    
    # 显示可能的问题数据
    print("\n检查缺失值:")
    print(data.isnull().sum())
    
    if 'gender' in data.columns:
        invalid_genders = data[~data['gender'].isin(['M', 'F'])]['gender'].unique()
        if len(invalid_genders) > 0:
            print("\n检测到非法性别值:")
            print(invalid_genders)

if __name__ == "__main__":
    # 生成100名学生的数据
    generate_student_data(100) 