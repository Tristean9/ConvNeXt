import json
import matplotlib.pyplot as plt

# 加载JSON数据
with open('log/final_all_results.json', 'r') as file:
    data = json.load(file)

# 提取键和值
keys = list(data.keys())[::-1]
values = list(data.values())[::-1]

# 将值转换为百分数形式，保留一位小数
values_percentage = [f"{value * 100:.1f}" for value in values]

# 创建条形图
plt.figure(figsize=(12, 8))
plt.barh(keys, values, color='#58508D')
plt.xlabel('Accuracy (%)')
plt.ylabel('Model')
plt.title('Bar Chart of Values')

# 在条形图上添加百分数标签
for i, value in enumerate(values):
    plt.text(value, i, values_percentage[i], ha='left', va='center')

# 保存图形
plt.savefig('log/result_chart.png')

# 显示图形
plt.show()
