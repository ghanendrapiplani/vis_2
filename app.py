import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import manifold
from sklearn.metrics import pairwise_distances

# First of all you have to import it from the flask module:
app = Flask(__name__)

base_cols = ['AUSTRALIA - AUSTRALIAN DOLLAR/US$', 'EURO AREA - EURO/US$',
             'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
             'UNITED KINGDOM - UNITED KINGDOM POUND/US$', 'BRAZIL - REAL/US$',
             'CANADA - CANADIAN DOLLAR/US$', 'CHINA - YUAN/US$',
             'HONG KONG - HONG KONG DOLLAR/US$', 'INDIA - INDIAN RUPEE/US$',
             'KOREA - WON/US$', 'MEXICO - MEXICAN PESO/US$',
             'SOUTH AFRICA - RAND/US$', 'SINGAPORE - SINGAPORE DOLLAR/US$',
             'DENMARK - DANISH KRONE/US$', 'JAPAN - YEN/US$',
             'MALAYSIA - RINGGIT/US$', 'NORWAY - NORWEGIAN KRONE/US$',
             'SWEDEN - KRONA/US$', 'SRI LANKA - SRI LANKAN RUPEE/US$',
             'SWITZERLAND - FRANC/US$', 'TAIWAN - NEW TAIWAN DOLLAR/US$',
             'THAILAND - BAHT/US$']

renamed_cols = ['AUSTRALIA_AUSDOLLARS', 'EUROPE_EURO', 'NEWZEALAND_NZDOLLAR',
                'UNITEDKINGDOM_POUND', 'BRAZIL_REAL', 'CANADA_CANADIANDOLLAR',
                'CHINA_YUAN', 'HONGKONG_HKDOLLAR', 'INDIA_INDIANRUPEE', 'KOREA_WON',
                'MEXICO_PESO', 'SOUTHAFRICA_RAND', 'SINGAPORE_SINGAPOREDOLLAR',
                'DENMARK_DANISHKRONE', 'JAPAN_YEN', 'MALAYSIZ_RINGGIT', 'NORWAY_NORWAYKRONE',
                'SWEDEN_SWEDISHKRONE', 'SRILANKA_SRILANKANRUPEE', 'SWITZERLAND_FRANC', 'TAIWAN_NEWTAIWANDOLLAR',
                'THAILAND_BAHT']

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/kmeans_scree_plot")
def kmeans():
    global df_main
    df = kmeans_cluster(df_main)
    return render_template("pca.html", data=pca(df),  title=json.dumps({"title": "PCA for KMeans Clustered Data"}))


@app.route("/fulldata_scree_plot")
def fulldata():
    global df_main
    return render_template("pca.html", data=pca(df_main) ,  title=json.dumps({"title": "PCA Scree Plot for unsampled data"}))


@app.route("/random_scree_plot")
def random():
    global df_main_random
    return render_template("pca.html", data=pca(df_main_random) ,  title=json.dumps({"title": "PCA Scree Plot for random sampled data"}))


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
    print(df_kcluster0.shape)
    print(df_kcluster1.shape)
    print(df_kcluster2.shape)
    df_ = pd.concat([df_kcluster0, df_kcluster1, df_kcluster2])
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
    marker = [0 for x in range(1, 11)]
    sum = 0
    index = -1
    for i in range(len(per_var)):
        sum += per_var[i]
        if (sum > 80):
            index = i
            break
    marker[index] = 1
    sp_df['marker'] = marker
    chart_data = sp_df.to_dict(orient='records')
    chart_data = json.dumps(chart_data)
    data = {'chart_data': chart_data}
    return data


def process_val(val):
    if val == "ND":
        return 0.0
    else:
        return float(val)


@app.route('/scatter_2d_fulldata')
def scatter_fulldata_plot():
    global df_main
    pca_data = PCA(n_components=2)
    pca_data.fit(df_main)
    df_ = pca_data.transform(df_main)
    data_final = pd.DataFrame(df_)
    loadings = np.sum(np.square(pca_data.components_), axis=0)
    indices = loadings.argsort()[-2:][::-1]
    print(indices)
    for i in range(len(indices)):
        data_final[df_main_random.columns[indices[i]]] = df_main_random[
            df_main_random.columns[indices[i]]]
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D PCA Scatter plot for full data for top 2 PCA"}))


@app.route('/scatter_2d_random')
def scatter_random_plot():
    global df_main_random
    pca_data = PCA(n_components=2)
    pca_data.fit(df_main_random)
    df_ = pca_data.transform(df_main_random)
    data_final = pd.DataFrame(df_)
    loadings = np.sum(np.square(pca_data.components_), axis=0)
    indices = loadings.argsort()[-2:][::-1]
    print(indices)
    for i in range(len(indices)):
        data_final[df_main_random.columns[indices[i]]] = df_main_random[
            df_main_random.columns[indices[i]]]
    print(data_final)

    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D PCA Scatter plot for random sampeled data for top 2 PCA"}))


@app.route('/scatter_2d_kmeans')
def scatter_kmeans_plot():
    global df_main_kmeans
    pca_data = PCA(n_components=2)
    pca_data.fit(df_main_kmeans)
    df_ = pca_data.transform(df_main_kmeans)
    data_final = pd.DataFrame(df_)

    loadings = np.sum(np.square(pca_data.components_), axis=0)
    indices = loadings.argsort()[-2:][::-1]
    print(indices)
    for i in range(len(indices)):
        data_final[df_main_kmeans.columns[indices[i]]] = df_main_kmeans[
            df_main_kmeans.columns[indices[i]]]
    print(data_final)
    data_final['clusterid'] = np.nan
    x = 0
    for index, row in df_main_kmeans.iterrows():
        data_final['clusterid'][x] = row['cluster_num']
        x = x + 1
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D PCA Scatter plot for kmeans clustered data for top 2 PCA"}))


@app.route('/mds_euclidean_full')
def mds_euclidean_full():
    global df_main
    df_full_subset = df_main[0:int(0.1 * len(df_main))]
    print("dfdf {}".format(df_full_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_full_subset, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS Euclidian distance on unsampled data for top 2 PCA"}))


@app.route('/mds_euclidean_random')
def mds_euclidean_random():
    global df_main_random
    df_random_subset = df_main_random[0:int(0.1 * len(df_main))]
    print("dfdf {}".format(df_random_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_random_subset, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS Euclidian distance on random data for top 2 PCA"}))


@app.route('/mds_euclidean_kmeans')
def mds_euclidean_kmeans():
    global df_main_kmeans
    df_kmeans_subset = df_main_kmeans.sample(int(0.2 * len(df_main_kmeans)))
    print("dfdf {}".format(df_kmeans_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_kmeans_subset, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    data_final['clusterid'] = np.nan
    x = 0
    for index, row in df_kmeans_subset.iterrows():
        data_final['clusterid'][x] = row['cluster_num']
        x = x + 1
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS Euclidian distance on Kmeans clustered data for top 2 PCA"}))


@app.route('/mds_correlation_full')
def mds_correlation_full():
    global df_main
    df_full_subset = df_main[0:int(0.1 * len(df_main))]
    print("dfdf {}".format(df_full_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_full_subset, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS correlation distance on unsampled data for top 2 PCA"}))


@app.route('/mds_correlation_random')
def mds_correlation_random():
    global df_main_random
    df_random_subset = df_main_random[0:int(0.1 * len(df_main))]
    print("dfdf {}".format(df_random_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_random_subset, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS correlation distance on random data for top 2 PCA"}))


@app.route('/mds_correlation_kmeans')
def mds_correlation_kmeans():
    global df_main_kmeans
    df_kmeans_subset = df_main_kmeans.sample(int(0.1 * len(df_main)))
    print("dfdf {}".format(df_kmeans_subset.shape))
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(df_kmeans_subset, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_final = pd.DataFrame(X)
    data_final['clusterid'] = np.nan
    x = 0
    for index, row in df_kmeans_subset.iterrows():
        data_final['clusterid'][x] = row['cluster_num']
        x = x + 1
    return render_template("scatter_2d.html", data=json.dumps(data_final.to_dict()), title=json.dumps({"title":"2D Scatter plot for MDS correlation distance on Kmeans clustered data for top 2 PCA"}))



@app.route('/scatter_matrix_full')
def scatter_matrix_full():
    global df_main
    data_columns = pd.DataFrame()
    pca = PCA()
    pca.fit(df_main)
    loadings = np.sum(np.square(pca.components_), axis=0)
    indices = loadings.argsort()[-3:][::-1]
    for i in range(len(indices)):
        data_columns[df_main.columns[indices[i]]] = df_main[df_main.columns[indices[i]]]
    return render_template("scatter_matrix.html",
                           rs=json.dumps({'rs': True}),
                           data=json.dumps(data_columns.to_dict()),  title=json.dumps({"title": "Scatter Matrix for Unsampled Data for top 3 PCA"}))


@app.route('/scatter_matrix_random')
def scatter_matrix_random():
    global df_main_random
    data_columns = pd.DataFrame()
    pca = PCA()
    pca.fit(df_main_random)
    loadings = np.sum(np.square(pca.components_), axis=0)
    indices = loadings.argsort()[-3:][::-1]
    for i in range(len(indices)):
        data_columns[df_main_random.columns[indices[i]]] = df_main_random[df_main_random.columns[indices[i]]]
    return render_template("scatter_matrix.html",
                           rs=json.dumps({'rs': True}),
                           data=json.dumps(data_columns.to_dict()),  title=json.dumps({"title": "Scatter Matrix Random Data for top 3 PCA"}))


@app.route('/scatter_matrix_kmeans')
def scatter_matrix_kmeans():
    global df_main_kmeans
    data_columns = pd.DataFrame()
    pca = PCA()
    pca.fit(df_main_random)
    loadings = np.sum(np.square(pca.components_), axis=0)
    indices = loadings.argsort()[-3:][::-1]
    for i in range(len(indices)):
        data_columns[df_main_random.columns[indices[i]]] = df_main_random[df_main_random.columns[indices[i]]]
    data_columns['clusterid'] = df_main_kmeans['cluster_num']
    return render_template("scatter_matrix.html",
                           rs=json.dumps({'rs': False}),
                           data=json.dumps(data_columns.to_dict()),  title=json.dumps({"title": "Scatter Matrix Kmeans Data for top 3 PCA"}))


if __name__ == "__main__":
    df = pd.read_csv('forex_rates.csv')
    df_main = df.drop("Time Serie", axis=1)
    df_main.columns = renamed_cols
    print(df_main.columns)
    for c in df_main.columns:
        df_main[c] = df_main[c].map(lambda val: process_val(val))
    scaler = StandardScaler()
    df_main[df_main.columns] = scaler.fit_transform(df_main[df_main.columns])
    df_main_random = df_main.sample(n=int(0.25 * len(df_main)))
    df_main_kmeans = plot_k(3, df_main)
    app.run(debug=True)
