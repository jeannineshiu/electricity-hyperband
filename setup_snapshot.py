from daytona import Daytona

daytona = Daytona()
sb = daytona.create()

# 把你的專案拉進去（需要 repo 是 public）
sb.process.exec("git clone https://github.com/YOUR/electricity-price-forecasting /workspace/project")
sb.process.exec("pip install lightgbm pandas scikit-learn pyarrow numpy")

# 驗證資料可以讀
resp = sb.process.exec("python -c \"import pandas as pd; df=pd.read_parquet('/workspace/project/data/processed/features_2020_2024.parquet'); print(len(df))\"")
print("Rows:", resp.result)

sb._experimental_create_snapshot("elec-forecast-v1")
sb.delete()
print("Snapshot ready.")
