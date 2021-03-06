import numpy as np
import random




def generateMustlinkCannotlink(X, Y):
    n = len(X)
    ml = np.zeros((n, n))
    cl = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if Y[i] == Y[j]:
                fuzzy_degree = random.uniform(0, 1)
                ml[i][j] = fuzzy_degree
                ml[j][i] = fuzzy_degree
                cl[i][j] = -1
                cl[j][i] = -1
            else:
                fuzzy_degree = random.uniform(0, 1)
                cl[i][j] = fuzzy_degree
                cl[j][i] = fuzzy_degree
                ml[i][j] = -1
                ml[j][i] = -1

    return ml, cl



def generateMustlinkCannotlink(X,Y):
    n = len(X)
    ml = np.zeros((n, n))
    cl = np.zeros((n, n))
    # generate two int random i and j
    for i in range(n):
        for j in range(n):
            if Y[i] == Y[j]:
                fuzzy_degree = random.uniform(0, 1)
                ml[i][j] = fuzzy_degree
                ml[j][i] = fuzzy_degree
                cl[i][j] = -1
                cl[j][i] = -1
            else:
                fuzzy_degree = random.uniform(0, 1)
                cl[i][j] = fuzzy_degree
                cl[j][i] = fuzzy_degree
                ml[i][j] = -1
                ml[j][i] = -1

    return ml,cl

def generateRandomConstraints(x, y, max_constraint=30):
    LINK_ARRAY_SIZE = max_constraint
    samples = np.random.choice(len(x), LINK_ARRAY_SIZE)
    must_links = []
    must_links_index = []
    ml_matrix = np.zeros((len(x), len(x)))
    cl_matrix = np.zeros((len(x), len(x)))
    ml_dic = {}
    cl_dic = {}
    for sample in samples:
        target_sample = y[sample]
        is_selected = [-1] * len(x)
        while -1 in is_selected:
            selected = np.random.choice(len(x), 1)[0]  # check selected not selected before
            if is_selected[selected] != -1:
                continue
            # new
            flag = 0
            for src, des in must_links_index:
                # flag = 0
                if src == sample and des == selected:
                    #print("must continue:", src, des)
                    flag = 1
                    break
            if flag:
                flag = 0
                continue
            # end new

            is_selected[selected] = 1
            if target_sample == y[selected]:
                if sample == selected:
                    continue
                ml = [np.asarray(x[sample]), np.asarray(x[selected])]
                ml_index = (sample, selected)
                must_links.append(ml)
                must_links_index.append(ml_index)

                degree = random.uniform(0, 1)
                ml_matrix[sample][selected] = degree
                ml_matrix[selected][sample] = degree
                ml_dic[ml_index]=degree
                break
            else:
                continue

    samples = np.random.choice(len(x), LINK_ARRAY_SIZE)  # DOIT
    cannot_links = []
    cannot_links_index = []
    for sample in samples:
        target_sample = y[sample]
        is_selected = [-1] * len(x)
        while -1 in is_selected:
            selected = np.random.choice(len(x), 1)[0]
            if is_selected[selected] != -1:
                continue
            # new
            flag = 0
            for src, des in cannot_links_index:
                flag = 0
                if src == sample and des == selected:
                    #print("cannot continue:", src, des)
                    flag = 1
                    break
            if flag:
                flag = 0
                continue
            # end new

            is_selected[selected] = 1

            if target_sample != y[selected]:
                cl = [np.asarray(x[sample]), np.asarray(x[selected])]
                cl_index = (sample, selected)
                cannot_links.append(cl)
                cannot_links_index.append(cl_index)

                degree = random.uniform(0, 1)
                cl_matrix[sample][selected] = degree
                cl_matrix[selected][sample] = degree
                cl_dic[cl_index] = degree
                break
            else:
                continue

    # TODO
    links = {'must_links_index': must_links_index, 'cannot_links_index': cannot_links_index}

    # np.save('best_'+dataset+str(LINK_ARRAY_SIZE), links)

    # m_c_links = np.load('C:\\Users\Amin\Documents\PycharmProjects\constrained_clustering\\best_iris30.npy').item()
    # ml_idx = m_c_links['must_links_index']
    # cl_idx = m_c_links['cannot_links_index']
    n = len(x)
    for i in range(n):
        for j in range(n):
            if ml_matrix[i][j] == 0.0:
                ml_matrix[i][j] = -1.0
            if cl_matrix[i][j]:
                cl_matrix[i][j] = -1.0

    return must_links_index, cannot_links_index, ml_matrix, cl_matrix, ml_dic, cl_dic

def transitive_entailment_graph(ml, cl, dslen):
    ml_graph = {}
    cl_graph = {}
    for i in range(dslen):
        ml_graph[i] = set()
        cl_graph[i] = set()

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    def dfs(v, graph, visited, component):
        visited[v] = True
        for j in graph[v]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(v)

    visited = [False] * dslen
    neighborhoods = []
    for i in range(dslen):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)
        for y in ml_graph[j]:
            cl_graph[i].add(y)
            cl_graph[y].add(i)
        for x in ml_graph[i]:
            cl_graph[x].add(j)
            cl_graph[j].add(x)
            for y in ml_graph[j]:
                cl_graph[x].add(y)
                cl_graph[y].add(x)

    return ml_graph, cl_graph, neighborhoods


