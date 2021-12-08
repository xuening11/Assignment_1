## Customer Segmentation and Attribution & Allocation
## Attribution_technical: entrance to sign-up form captured by product (Last touchpoint)
## Attribution_survey: customer reported 'where did you hear about us' (First touchpoint)

## Part 1 Customer Segmentation
##Age, Package Type, Intended Use, Preferred Genre
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import datetime

pd.set_option('display.max_columns', 25)


# ------ Define functions ------
def run_kmeans(n_clusters_f, init_f, df_f):
    ##### Complete this function
    # This function should at least take a dataframe as an argument. I have suggested additional arguments you may
    # want to provide, but these can be changed as you need to fit your solution.
    # The output of this function should be the input data frame will the model object KMeans and a data summary. The
    # function will need to add an additional column to the input dataframe called 'predict_cluster_kmeans'
    # that contains the cluster labels assigned by the algorithm.

    k_means_model_f = KMeans(n_clusters=n_clusters_f, init=init_f).fit_predict(df_f)
    k_means_model_f = pd.DataFrame(k_means_model_f)
    k_means_model_f['123'] = df_f.index
    k_means_model_f = k_means_model_f.set_index('123', drop=True, append=False, inplace=False, verify_integrity=False)
    k_means_model_f.columns = ['predict_cluster_kmeans']

    df_f = pd.concat([df_f, k_means_model_f], axis=1, join='outer')


    # summarize cluster attributes
    k_means_model_f_summary = df_f.groupby('predict_cluster_kmeans').agg(attribute_summary_method_dict)
    return k_means_model_f, k_means_model_f_summary

# ------ Import data ------
df_cluster = pd.read_csv('subscribers_cleaned_dummified_1.csv')

# ------ RUN CLUSTERING -----
# --- set parameters
n_clusters = 3
init_point_selection_method = 'k-means++'

# --- split to test and train
df_cluster_train, df_cluster_test, _, _, = train_test_split(df_cluster, [1]*df_cluster.shape[0], test_size=0.33)   # ignoring y values for unsupervised

# --- fit model
attribute_summary_method_dict = {'weekly_consumption_hour': np.mean, 'age': np.mean,'package_type_base':sum,'package_type_economy':sum,'package_type_enhanced':sum,'preferred_genre_comedy':sum,'preferred_genre_drama':sum,'preferred_genre_international':sum,'preferred_genre_other':sum,'preferred_genre_regional':sum,'intended_use_access to exclusive content':sum,'intended_use_education':sum,'intended_use_expand international access':sum,'intended_use_expand regional access':sum,'intended_use_other':sum,'intended_use_replace OTT':sum,'intended_use_supplement OTT':sum,'male_TF_False':sum,'male_TF_True':sum}
col_output_order = ['weekly_consumption_hour', 'age', 'package_type_base',
       'package_type_economy', 'package_type_enhanced',
       'preferred_genre_comedy', 'preferred_genre_drama',
       'preferred_genre_international', 'preferred_genre_other',
       'preferred_genre_regional', 'intended_use_access to exclusive content',
       'intended_use_education', 'intended_use_expand international access',
       'intended_use_expand regional access', 'intended_use_other',
       'intended_use_replace OTT', 'intended_use_supplement OTT',
       'male_TF_False', 'male_TF_True'] ##### specify order of output columns for easy of readability

# training data
train_model, train_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_train.reindex())
# testing data
test_model, test_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_test.reindex())
# all data
model, model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)

# --- run for various number of clusters
##### add the code to run the clustering algorithm for various numbers of clusters
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(df_cluster)
    inertias.append(model.inertia_)

# --- draw elbow plot
##### create an elbow plot for your numbers of clusters in previous step
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# --- export output
model = KMeans(n_clusters=3, n_init=10)
model.fit(df_cluster)
output = pd.DataFrame(model.cluster_centers_)
output.columns = df_cluster.columns
output.to_csv('Kmeans_output.csv')


