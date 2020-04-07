import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random

# First of all you have to import it from the flask module:
app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    pd.set_option('display.max_columns', None)
    global df
    # The current request method is available by using the method attribute
    if request.method == 'POST':
        # if request.form['data'] == 'received':
        data = df[['date', 'open']]
        data = data.rename(columns={'open': 'close'})
        print(data)
        print("Hello World!")
        chart_data = data.to_dict(orient='records')
        chart_data = json.dumps(chart_data, indent=2)
        data = {'chart_data': chart_data}
        return jsonify(data)  # Should be a json string

    data = df[['date', 'close']]
    chart_data = data.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)


@app.route("/kmeans", methods=['POST', 'GET'])
def kmeans():
    global df_main
    df = kmeans_cluster(df_main)
    return render_template("pca.html", data=pca(df))


@app.route("/fulldata", methods=['POST', 'GET'])
def fulldata():
    global df_main
    return render_template("pca.html", data=pca(df_main))


def plot_k(k, x):
    df_ = x
    kmeans_ = KMeans(n_clusters=k)
    y = kmeans_.fit_predict(df_)
    df_['cluster_num'] = y
    test = {}
    for i in y:
        try:
            test[i] = test[i] + 1
        except Exception:
            test[i] = 1
    print("cluster stats = cluster_num:nodes {}".format(test))

    kcluster0 = df_[df_['cluster_num'] == 0]
    kcluster1 = df_[df_['cluster_num'] == 1]
    kcluster2 = df_[df_['cluster_num'] == 2]
    df_kcluster0 = kcluster0.sample(n=int(0.25 * len(kcluster0)))
    df_kcluster1 = kcluster1.sample(n=int(0.25 * len(kcluster1)))
    df_kcluster2 = kcluster2.sample(n=int(0.25 * len(kcluster2)))
    # print(df_kcluster0.shape)
    # print(df_kcluster1.shape)
    # print(df_kcluster2.shape)
    df_ = pd.concat([df_kcluster0, df_kcluster1, df_kcluster2])
    # print(df_)
    return df_


def kmeans_cluster(x):
    k_arr = range(1, 10)
    x_zip = x
    inertias = []
    mapping_inert = {}
    for k in k_arr:
        kmeanModel = KMeans(n_clusters=k).fit(x_zip)
        inertia = kmeanModel.inertia_
        inertias.append(inertia)
        mapping_inert[k] = inertia

    print("inertias {} ".format(inertias))
    plt.plot(k_arr, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertias')
    plt.title('The Elbow Method using Inertia')
    plt.show()
    return plot_k(3, x_zip)


def find_area(val):
    area = 150
    try:
        val_arr = val.replace(" Sq.ft.", "").split(" X ")
        area = float(val_arr[0]) * float(val_arr[1])
    except Exception as e:
        pass
    return area


def hot_encoding(df_, col_list):
    for col in col_list:
        df_ = df_.merge(pd.get_dummies(df_[col]), left_index=True, right_index=True)
        df_ = df_.drop([col], axis=1)
    return df_


# def pca(df_, p_c):
#     print("df shape rows {} cols {}".format(df_.shape[0], df_.shape[1]))
#     x = StandardScaler().fit_transform(df_)
#     pca_ = decomposition.PCA(n_components=p_c)
#     pc = pca_.fit_transform(x)
#     pc_list = []
#     for f in range(p_c):
#         pc_list.append("PC{}".format(f+1))
#     pc_df = pd.DataFrame(data=pc, columns=pc_list)
#     pc_df['Cluster'] = df_['cluster_num']
#     print("{}".format(pc_df.head()))
#     df_pca = pd.DataFrame({'var': pca_.explained_variance_ratio_, 'PC': pc_list})
#     sns.barplot(x='PC', y="var", data=df_pca, color="c")
#     plt.show()


@app.route("/random", methods=['GET'])
def random_sampling():
    global df_main
    print(len(df_main))
    df_random = df_main.sample(n=int(0.25 * len(df_main)))
    print("len of random = {}".format(len(df_random)))
    return render_template("pca.html", data=pca(df_random))


def pca(df_=None):
    pca_obj = PCA()
    scaled_data = preprocessing.scale(df_)
    pca_obj.fit(scaled_data)
    per_var = np.round(pca_obj.explained_variance_ratio_ * 100, decimals=1).tolist()[0:10]
    labels = ['PC' + str(x) for x in range(1, 11)]
    sp_df = pd.DataFrame({'label': labels, 'eigenvalue': per_var})
    sp_df['cumulative_eigenvalue'] = sp_df['eigenvalue'].cumsum()
    chart_data = sp_df.to_dict(orient='records')
    chart_data = json.dumps(chart_data)
    data = {'chart_data': chart_data}
    return data


def process_val(val):
    if val == "ND":
        return 0.0
    else: return float(val)


if __name__ == "__main__":
    df = pd.read_csv('forex_rates.csv')
    df_main = df.drop("Time Serie", axis=1)
    for c in df_main.columns:
        df_main[c] = df_main[c].map(lambda val: process_val(val))


    # df_main = df_main.replace()
    # df_main['room_area'] = df_main['room_area'].map(lambda area: find_area(area))
    # df_main = pd.read_csv('goibibo_data.csv')
    # df_main['room_area'] = df_main['room_area'].map(lambda area: find_area(area))
    # cat_cols = ['hotel_category', 'property_type', 'additional_info', 'address', 'area', 'city', 'country',
    #             'crawl_date', 'hotel_brand', 'hotel_description',
    #             'hotel_facilities', 'latitude', 'locality', 'longitude', 'pageurl', 'point_of_interest', 'property_id',
    #             'property_name', 'province', 'qts', 'query_time_stamp', 'review_count_by_category', 'room_facilities',
    #             'room_type', 'similar_hotel',
    #             'site_stay_review_rating', 'sitename', 'state', 'uniq_id']
    #
    # for c in cat_cols:
    #     print(c)
    #     df_main = df_main.drop(c, axis=1)
    # print("df new ")
    # print(df_main)
    # df_new = process_csv('goibibo_data.csv')
    # # df_main = hot_encoding(df_main, cat_cols)
    app.run(debug=True)
