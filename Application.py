import Dataset
from Tasks import TaskData
from MLPipeline import Pipeline

app = Pipeline()

@app.Dataset()
def AllData_v2():
    return TaskData(Dataset.AllData_v2, "AllData_v2")