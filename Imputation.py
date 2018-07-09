# import pandas as pd
# import Dataset
# import seaborn as sb
# import matplotlib.pyplot as plt

# # train, test, label = Dataset.Load("AllData_v3")

# # train["TARGET"] = label

# train = pd.read_csv("../input/application_train.csv")
# test = pd.read_csv("../input/application_test.csv")

# # # subset = train.iloc[:,0:50]

# # # corr = subset.corr()

# # corr = train.corr()

# # sb.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
# # plt.show()


# # a = pd.DataFrame(train.corr()["TARGET"].sort_values(ascending=False))
# # a.to_csv("../corr_apptrain.csv")




# del_cols = [
# "CNT_FAM_MEMBERS",
# "OBS_30_CNT_SOCIAL_CIRCLE",
# "OBS_60_CNT_SOCIAL_CIRCLE",
# "REG_REGION_NOT_WORK_REGION",
# "REG_REGION_NOT_LIVE_REGION",
# "FLAG_DOCUMENT_2",
# "FLAG_DOCUMENT_21",
# "LIVE_REGION_NOT_WORK_REGION",
# "AMT_REQ_CREDIT_BUREAU_DAY",
# "AMT_REQ_CREDIT_BUREAU_HOUR",
# "AMT_REQ_CREDIT_BUREAU_WEEK",
# "FLAG_MOBIL",
# "FLAG_CONT_MOBILE",
# "FLAG_DOCUMENT_20",
# "FLAG_DOCUMENT_5",
# "FLAG_DOCUMENT_12",
# "FLAG_DOCUMENT_19",
# "FLAG_DOCUMENT_10",
# "FLAG_DOCUMENT_7",
# "NONLIVINGAPARTMENTS_MODE",
# "FLAG_EMAIL",
# "AMT_REQ_CREDIT_BUREAU_QRT",
# "SK_ID_CURR",
# "FLAG_DOCUMENT_4",
# "NONLIVINGAPARTMENTS_MEDI",
# "NONLIVINGAPARTMENTS_AVG",
# "FLAG_DOCUMENT_17",
# "AMT_INCOME_TOTAL",
# "FLAG_DOCUMENT_11",
# "FLAG_DOCUMENT_9",
# "FLAG_DOCUMENT_15",
# "FLAG_DOCUMENT_18",
# "FLAG_DOCUMENT_8",
# "YEARS_BEGINEXPLUATATION_MODE",
# "FLAG_DOCUMENT_14",
# "YEARS_BEGINEXPLUATATION_AVG",
# "YEARS_BEGINEXPLUATATION_MEDI",
# "LANDAREA_MODE",
# "LANDAREA_AVG",
# "LANDAREA_MEDI",
# "FLAG_DOCUMENT_13",
# "FLAG_DOCUMENT_16",
# "AMT_REQ_CREDIT_BUREAU_MON",
# "NONLIVINGAREA_MODE",
# "AMT_ANNUITY",
# "NONLIVINGAREA_MEDI",
# "NONLIVINGAREA_AVG",
# "COMMONAREA_MODE",
# "ENTRANCES_MODE",
# "COMMONAREA_AVG",
# "COMMONAREA_MEDI",
# "ENTRANCES_MEDI",
# "ENTRANCES_AVG",
# "BASEMENTAREA_MODE"]


# for i in del_cols:
#     print("deleting column: " + i)
#     del train[i]
#     del test[i]

# from sklearn.preprocessing import Imputer

# print("initializing imputer")
# imp = Imputer(strategy="knn", n_neighbors=5, axis=0)
# print("transforming imputer")
# a = imp.fit_transform(train)
# print("writing to file")

# pd.DataFrame(a).to_csv("../imputed.csv")




class a(object):
    def __init__(self, v):
        self._v_ = v

    def aa(self):
        print(self._v_)
        self.b()

    def b(self):
        pass


class b(a):
    def __init__(self, v):
        super().__init__(v)

    def aa(self):
        super(b,self).aa()
        print("aab")

    def b(self):
        print("bbb")


b("aazaaaaa").aa()
