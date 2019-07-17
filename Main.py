import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import copy
from load_datasets import load_dataset
import GenerateConstraint
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from COP_Kmeans import COP_KMeans

start = time.time()

DATASETS = ['iris',  # 0 *
            'glass',  # 1 *
            'wine',  # 2 *
            'pendigits', # 3
            'vowel']  # 4

# DATASET_NAME = ''
# DATASET_NUM = 0


performances_for_any_dataset = {}
for i in range(0, 3):
    DATASET_NAME = ''
    DATASET_NUM = i

    DATASET_NAME = DATASETS[DATASET_NUM]
    X, Y, k = load_dataset(DATASET_NAME)
    print("for "+DATASET_NAME)


    # remove duplicate data from some datasets
    unique_X, return_indexes = np.unique(X, axis=0, return_index=True)
    unique_Y = np.zeros(len(unique_X))
    ii = 0
    for index in return_indexes:
        unique_Y[ii] = Y[index]
        ii += 1

    X = unique_X
    Y = unique_Y

    # end remove duplicates

    linkage_matrix = linkage(X, 'ward', optimal_ordering=True)


    # pair_dis = np.array([17, 21, 31, 23, 30, 34, 21, 28, 39, 43])
    # ZZ = linkage(pair_dis, method="average", optimal_ordering=True)
    # linkage_matrix = ZZ


    ''' plot dendrogram '''

    def fancy_dendrogram(*args, **kwargs):
        global DATASET_NAME
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram for '+ DATASET_NAME)
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata


    fancy_dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        annotate_above=1,  # useful in small plots so annotations don't overlap
    )
    plt.show()

    l_matrix_int = linkage_matrix[:, [0, 1, 3]].astype(int)
    l_matrix_float = linkage_matrix[:, [2]]


    # N = 5
    # count = N-1

    # *** create ultrametric distance matrix from dendrogram then create fuzzy similarity matrix

    N = len(X)
    count = len(X) - 1
    dic = {}

    ultrametric = np.zeros((N, N))

    for i in range(len(linkage_matrix)):
        a = l_matrix_int[i][0]
        b = l_matrix_int[i][1]
        value = l_matrix_float[i][0]
        # observation = l_matrix_int[i][2]
        count = count + 1

        if a < N and b < N:
            ultrametric[a][b] = value
            ultrametric[b][a] = value
            dic[count] = {a, b}

        if a < N <= b:
            set_b = dic[b]
            for j in set_b:
                ultrametric[a][j] = value
                ultrametric[j][a] = value

            set_b.add(a)
            dic[count] = set_b

        if a >= N > b:
            set_a = dic[a]
            for j in set_a:
                ultrametric[b][j] = value
                ultrametric[j][b] = value

            set_a.add(b)
            dic[count] = set_a

        if a >= N and b >= N:
            set_a = dic[a]
            set_b = dic[b]
            dic[count] = set_a.union(set_b)

            for a_i in set_a:
                for b_j in set_b:
                    ultrametric[a_i][b_j] = value
                    ultrametric[b_j][a_i] = value

    # for i in range(N):
    #     ultrametric[i][i] = 1.0

    max_dis = np.max(ultrametric)
    for i in range(N):
        for j in range(N):
            ultrametric[i][j] = 1 - (ultrametric[i][j] / max_dis)
            if ultrametric[i][j] == 0.0:
                ultrametric[i][j] += 0.1

    #########################################

    #########################################
    fuzzy_similarity_matrix = copy.deepcopy(ultrametric)

    # unique
    alpha_cut_set = np.unique(ultrametric)
    alpha_cut_set = np.sort(alpha_cut_set)

    size_alpha_cut_set = len(alpha_cut_set)
    temp_alpha_cut_set = np.arange(0, 1, 1.0/size_alpha_cut_set)
    temp_alpha_cut_set[-1] = 1.0

    # change fuzzy similarity matrix with range of alpha cut set
    for i in range(size_alpha_cut_set):
        fuzzy_similarity_matrix = np.where(fuzzy_similarity_matrix == alpha_cut_set[i], temp_alpha_cut_set[i],
                                           fuzzy_similarity_matrix)


    ############################################################################

    # constraint for all pair
    '''
    from paper :
    Instance level constraints have been generated according to
    the class labels of the datasets. For that purpose, 
    a fuzzy mustlink constraint has been created for ""every pair"" of instances
    belonging to the same class. The degree of belief on that
    constraint, β, has been assigned randomly.
    '''
    ml, cl = GenerateConstraint.generateMustlinkCannotlink(X, Y)

    '''
    generate constraint with number 
    '''
    Must_links_, Cannot_links_, ml_matrix, cl_matrix, ml_dic, cl_dic = GenerateConstraint.generateRandomConstraints(X, Y, 40)
    # ml = ml_matrix
    # cl = cl_matrix

    # set weights
    v = 1
    w = -0.5

    n_sample = len(X)

    alpha_values = []
    max_h_alpha = -1
    opt_alpha = -1

    for alpha in alpha_cut_set:
        c_alpha = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                if fuzzy_similarity_matrix[i][j] >= alpha:
                    c_alpha[i][j] = 1

        h_alpha = 0
        for i in range(n_sample):
            for j in range(n_sample):
                if ml[i][j] >= alpha and c_alpha[i][j] == 1:
                    h_alpha += v
                elif ml[i][j] >= alpha and c_alpha[i][j] == 0:
                    h_alpha += w
                elif cl[i][j] >= alpha and c_alpha[i][j] == 0:  # ???? TODO
                    h_alpha += v
                elif cl[i][j] >= alpha and c_alpha[i][j] == 1:  # ???? TODO
                    h_alpha += w

        if max_h_alpha <= h_alpha:
            max_h_alpha = h_alpha
            opt_alpha = alpha

        alpha_values.append((alpha, h_alpha))



    ###############################################################################
    ###############################################################################
    """             Fuzzy Entropy For Optimum  number of constraints            """
    ###############################################################################
    ###############################################################################

    def Fuzzy_Entropy_For_Constraints(X,Y):
        number_of_constraint = np.arange(0, 2*len(X)+10, 10)
        number_of_constraint[0] = 1
        purity_result_of_constraint = {}
        for num_constraint in number_of_constraint:
            _, _, ml_matrix, cl_matrix, ml_dic, cl_dic = GenerateConstraint.generateRandomConstraints(X, Y, num_constraint)

            # fuzzy entropy for MustLink constraints
            size_of_ml_constraints = len(ml_dic)
            ml_sum = 0
            for key, membership_value in ml_dic.items():
                ml_sum += membership_value * math.log10(membership_value) + (1 - membership_value) * math.log10(1 - membership_value)

            ml_Entropy = -1 * (1.0 / size_of_ml_constraints) * ml_sum

            # fuzzy entropy for CannotLink constraints
            size_of_cl_constraints = len(cl_dic)
            cl_sum = 0
            for key, membership_value in cl_dic.items():
                cl_sum += membership_value * math.log10(membership_value) + (1 - membership_value) * math.log10(1 - membership_value)

            cl_Entropy = -1 * (1.0 / size_of_cl_constraints) * cl_sum

            # print(num_constraint, ml_Entropy, cl_Entropy)
            purity_result_of_constraint[num_constraint] = (ml_Entropy, cl_Entropy)
        return purity_result_of_constraint

    ###############################################################
    """ plot for Entropy of constraints average of 5 executions """
    ###############################################################
    entropy_list = []
    for i in range(5):
        entropy_list.append(Fuzzy_Entropy_For_Constraints(X, Y))

    entropy_result_dic = {}
    for key,_ in entropy_list[0].items():
        sum_purity_must_link = 0
        sum_purity_cannot_link = 0
        for i in range(5):
            sum_purity_must_link += entropy_list[i][key][0]
            sum_purity_cannot_link += entropy_list[i][key][1]
        entropy_result_dic[key] = (sum_purity_must_link / 5.0, sum_purity_cannot_link / 5.0)


    def plot_purity_constraints(purity_dic, DATASET_NAME):
        purity_result_sort = sorted(purity_dic.items(), key=lambda kv: (kv[0]))
        x_plot = []
        y_ml_plot = []
        y_cl_plot = []
        for i in range(len(purity_result_sort)):
            x_plot.append(purity_result_sort[i][0])
            y_ml_plot.append(purity_result_sort[i][1][0])  # for mulst-link
            y_cl_plot.append(purity_result_sort[i][1][1])  # for cannot-link


        # must-link plot
        plt.plot(x_plot, y_ml_plot, 'g-^')
        plt.title("Entropy of Must-Link Constraints \n"+DATASET_NAME.upper()+" dataset")
        plt.ylabel('Entropy')
        plt.xlabel('# of Constraints')
        plt.show()

        # cannot-link plot
        plt.plot(x_plot, y_cl_plot, 'g-^')
        plt.title("Entropy of Cannot-Link Constraints \n"+DATASET_NAME.upper()+" dataset")
        plt.ylabel('Entropy')
        plt.xlabel('# of Constraints')
        plt.show()


    plot_purity_constraints(entropy_result_dic, DATASET_NAME)


    ################################################################
    """      PART 2 - find Optimum alpha-cut and clustering      """
    ################################################################
    list_of_ARS = []
    list_of_NMI = []
    list_of_Purity = []


    y_true = Y
    # function definition
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


    '''---------------- COP-Kmeans ---------------'''

    ml_g, cl_g, _ = GenerateConstraint.transitive_entailment_graph(Must_links_, Cannot_links_, len(X))
    cop_kmeans = COP_KMeans(k, ml_g, cl_g)
    cop_kmeans.fit(X)
    y_pred_cop_kmeans = cop_kmeans.is_clustered

    # NMI cop_kmeans
    NMI_copkmeans = normalized_mutual_info_score(y_true, y_pred_cop_kmeans, average_method="arithmetic")
    ARS_copkmeans = adjusted_rand_score(y_true, y_pred_cop_kmeans)
    purity_copkmeans = purity_score(y_true, y_pred_cop_kmeans)

    '''---------------- simple-Kmeans ---------------'''
    k_means = KMeans(n_clusters=k).fit(X)
    y_pred_kmeans = k_means.labels_

    # NMI simple K-means
    NMI_kmeans = metrics.normalized_mutual_info_score(y_true, y_pred_kmeans, average_method="arithmetic")
    ARS_kmeans = adjusted_rand_score(y_true, y_pred_kmeans)
    purity_kmeans = purity_score(y_true, y_pred_kmeans)


    '''---------------- Hierarchical ----------------'''
    # hierarchical_clus = AgglomerativeClustering(n_clusters=k)
    # hierarchical_clus.fit(X)
    # y_pred_hierarchy = hierarchical_clus.labels_
    #
    #
    # NMI_hierarchy = metrics.normalized_mutual_info_score(y_true, y_pred_hierarchy, average_method="arithmetic")
    # ARS_hierarchy = adjusted_rand_score(y_true, y_pred_hierarchy)
    # purity_hierarchy = purity_score(y_true, y_pred_hierarchy)

    '''---------------- Fuzzy HSS (this paper) ----------------'''

    def get_cluster_labels(fuzzy_sim_matrix, alpha_cut):
        n = fuzzy_sim_matrix.shape[0]
        for i in range(n):
            for k in range(n):
                if fuzzy_sim_matrix[i][k] >= alpha_cut:
                    fuzzy_sim_matrix[i][k] = 1
                else:
                    fuzzy_sim_matrix[i][k] = 0

        clusters_dic = {}
        temp_list = []

        for m in range(n):
            if m in temp_list:
                continue
            for k in range(n):
                if fuzzy_sim_matrix[m][k] == 1:
                    if k not in temp_list:
                        temp_list.append(k)
                        if m in clusters_dic:
                            clusters_dic[m].append(k)
                        else:
                            clusters_dic[m] = [k]

        return clusters_dic

    clusters_labels = get_cluster_labels(fuzzy_similarity_matrix, opt_alpha)

    my_labels = np.zeros(len(X))
    for key,value in clusters_labels.items():
        my_labels[value] = key

    y_pred_SHH = my_labels


    NMI_fuzzy_HSS = metrics.normalized_mutual_info_score(y_true, y_pred_SHH, average_method="arithmetic")
    ARS_fuzzy_HSS = adjusted_rand_score(y_true, y_pred_SHH)
    purity_fuzzy_HSS = purity_score(y_true, y_pred_SHH)

    ################
    # add to lists #
    ################

    list_of_NMI.append(('cop',NMI_copkmeans))
    list_of_NMI.append(('kmeans', NMI_kmeans))
    list_of_NMI.append(('FHSS', NMI_fuzzy_HSS))

    list_of_ARS.append(('cop',ARS_copkmeans))
    list_of_ARS.append(('kmeans', ARS_kmeans))
    list_of_ARS.append(('FHSS', ARS_fuzzy_HSS))

    list_of_Purity.append(('cop',purity_copkmeans))
    list_of_Purity.append(('kmeans', purity_kmeans))
    list_of_Purity.append(('FHSS', purity_fuzzy_HSS))

    performances_for_any_dataset[DATASET_NAME] = [list_of_NMI, list_of_ARS, list_of_Purity]


#############################################################
"""           plot for performance of clustering          """
#############################################################
barWidth = 0.1
performance_methods = ['Purity', 'Normalized Mutual Information', 'Adjusted Rand Index']
for i in range(len(performance_methods)):
    bars1 = []
    bars2 = []
    bars3 = []
    x_labels = []
    for key, value in performances_for_any_dataset.items():
        # value[0] equal NMI
        # value[1] equal ARS
        # value[2] equal purity
        # value[i][j] : j is type of clustering (i=0 : cop-kmeans , i=1 : kmeans , i=2 : FHSS)

        bars1.append(value[i][0][1])
        bars2.append(value[i][1][1])
        bars3.append(value[i][2][1])

        x_labels.append(key)

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # # set height of bar
    # bars1 = [12, 30, 1, 8, 22]  # fuzzy HSS
    # bars2 = [28, 6, 16, 5, 10]  # cop-kmeans
    # bars3 = [29, 3, 24, 25, 17] # kmeans
    #
    # # Set position of bar on X axis
    #
    # r1 = np.arange(len(bars1))
    # r2 = [x + barWidth for x in r1]
    # r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#F44336', width=barWidth, edgecolor='white', label='COP k-means')
    plt.bar(r2, bars2, color='#4CAF50', width=barWidth, edgecolor='white', label='k-means')
    plt.bar(r3, bars3, color='#00BCD4', width=barWidth, edgecolor='white', label='Fuzzy HSS')

    # Add xticks on the middle of the group bars
    plt.xlabel('dataset', fontweight='bold')
    plt.ylabel(performance_methods[i])
    plt.xticks([r + barWidth for r in range(len(bars1))], x_labels)

    # Create legend & Show graphic
    plt.legend()
    plt.show()

end = time.time()

print(end-start)


# {'wine': [[('cop', 0.36575632485476905), ('kmeans', 0.4287568597645354), ('FHSS', 0.30541766011249055)], [('cop', 0.3650219160593678), ('kmeans', 0.37111371823084754), ('FHSS', 0.09207815295133594)], [('cop', 0.702247191011236), ('kmeans', 0.702247191011236), ('FHSS', 0.7303370786516854)]], 'iris': [[('cop', 0.7794851201966955), ('kmeans', 0.7621245181084993), ('FHSS', 0.549664659033792)], [('cop', 0.76550582591247), ('kmeans', 0.7365165921513978), ('FHSS', 0.3290482568384849)], [('cop', 0.9115646258503401), ('kmeans', 0.8979591836734694), ('FHSS', 0.9387755102040817)]], 'glass': [[('cop', 0.41157211377334774), ('kmeans', 0.4151599300090336), ('FHSS', 0.41985260298629457)], [('cop', 0.25153641755388456), ('kmeans', 0.26805899332949806), ('FHSS', 0.18857078969644142)], [('cop', 0.5539906103286385), ('kmeans', 0.5868544600938967), ('FHSS', 0.7136150234741784)]]}

