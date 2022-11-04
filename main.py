# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).round()
    low_limit = (quartile1 - 1.5 * interquantile_range).round()
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



col_list= ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in col_list:
    replace_with_thresholds(df, col)


col_list= ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in col_list:
    outlier_thresholds(df, col)

df.shape
df.count()
df.describe().T


df["of_on_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["of_on_total_price_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


df.dtypes

date_colums =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[date_colums] = df[date_colums].apply(pd.to_datetime)

df.info()

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)



cltv_df= pd.DataFrame({"customer_id": df["master_id"],
             "recency": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T": ((today_date - df["first_order_date"]).astype("timedelta64[D]"))/7,
             "frequency": df["of_on_total_ever"],
             "monetary": df["of_on_total_price_ever"] / df["of_on_total_ever"]})

cltv_df.dtypes
cltv_df.head()
cltv_df.tail()


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


plot_period_transactions(bgf)
plt.show(block=True)


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'].astype(int), cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])



cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)


cltv_df.sort_values("cltv", ascending=False).head(20)

cltv_df["segment_6_month"].shape

cltv_df["segment_6_month"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()
