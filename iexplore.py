# ============= inspect dataset ================
import pandas as pd
import os
import matplotlib.pyplot as plt
print('pandas imported successfully')

# ============= Dataset path ===================
#C:\Users\LENOVO\Documents\Projects\Bridge\AI\Anomaly\data
ROOT = os.path.expanduser('~')
data_path = os.path.join(ROOT, 'Documents', 'Projects', 'Bridge', 'AI', 'Anomaly', 'data', 'CICIDS_Flow.parquet')

# ============= import dataset =================
try:
    df = pd.read_parquet(data_path)
    print('Dataset loaded successfully')
except Exception as e:
    print(f'Error handling dataset {e}')
    
# ============== inspect dataset ===============    
print('Dataset Shape:', df.shape)
print('Dataset Columns:', df.columns)
print('Dataset Head:\n', df.head())
print('Dataset Info:\n', df.info())
print('Dataset Description:\n', df.describe())

# ============== missing values ===============
print('Missing Values Count:\n', df.isnull().sum())    

# ============= Explore Attack Label Distribution ===================
print('Attack Label Distribution:\n', df['attack_label'].value_counts())

# ============= Visualize Attack Label Distribution ===================
plt.figure(figsize=(10, 6))
df['attack_label'].value_counts().plot(kind='bar')
plt.title('Distribution of Attack Labels')
plt.xlabel('Attack Label')
plt.ylabel('Count')
plt.show()

# ============= Visualize Flow Duration Distribution ===================
plt.figure(figsize=(10, 6))
df['Flow Duration'].plot.hist(bins=50)
plt.title('Distribution of Flow Duration')
plt.xlabel('Flow Duration')
plt.ylabel('Count')
plt.show()
  