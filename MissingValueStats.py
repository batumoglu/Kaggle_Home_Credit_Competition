import pandas as pd

pd.options.mode.use_inf_as_na = True

application_train       = "../input/application_train.csv"
application_test        = "../input/application_test.csv"
bureau                  = "../input/bureau.csv"
bureau_balance          = "../input/bureau_balance.csv"
credit_card_balance     = "../input/credit_card_balance.csv"
installments_payments   = "../input/installments_payments.csv"
POS_CASH_balance        = "../input/POS_CASH_balance.csv"
previous_application    = "../input/previous_application.csv"

files = [application_train,
         application_test,
         bureau,
         bureau_balance,
         credit_card_balance,
         installments_payments,
         POS_CASH_balance,
         previous_application]



def find_empty_columns(df):   
        nan_cols = []
        a = df.isna().any()
        nan_cols.extend(a[a==True].index.values)
        # print(f + " : " + str(len(nan_cols)) + "/" + str(len(d.columns)))
        return nan_cols

def find_empty_percentage(df):
    stats = []
    nan_cols = find_empty_columns(df)
    for col in nan_cols:
        b = df[col].isna().sum()
        c = b / len(df.index) * 100
        stats.append((col, b, len(df.index), c))
        # print(col + ": \t\t" + str(b) + "/" + str(len(df.index)) + " [" + "%.2f" % c + "%]")
    stats_sorted = sorted(stats, key=lambda x:x[3])
    return stats_sorted

# for f in files:
#     print(f)
#     d = pd.read_csv(f)
#     e = find_empty_percentage(d)
#     for s in e:
#         print(s[0] + "\t\t: " + str(s[1]) + "/" + str(s[2]) + " [" + '%.2f' % s[3] + "%]")

d = pd.read_csv("../input/AllData_v4.train")
l = pd.read_csv("../input/AllData_v4.label")

e = find_empty_percentage(d)
missing_ratios = []
index = []
for s in e:
    print(s[0] + "\t\t: " + str(s[1]) + "/" + str(s[2]) + " [" + '%.2f' % s[3] + "%]")
    missing_ratios.append(s[3])
    index.append(s[0])

missing_df = pd.DataFrame(missing_ratios, columns=["Missing_Ratio %"], index=index)

d["TARGET"] = l

df_corr = d.corr()

for idx, val in pd.Series.iteritems(df_corr["TARGET"]):
    missing_df.set_value(idx, "Corr_Target",val)

missing_df.to_csv("AllData_v4_missing_corr.csv")

print(missing_df)

