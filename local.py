import itertools
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score, precision_score
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as aucfunc

# 读取txt文件，每行代表一条边
edges_file = "0.edges"

NetData = np.loadtxt(edges_file)
def Data_Shape(Data):
    List_A = []
    List_B = []
    for row in range(Data.shape[0]):
        List_A.append(Data[row][0])
        List_B.append(Data[row][1])
    List_A = list(set(List_A))
    List_B = list(set(List_B))
    length_A = len(List_A)
    length_B = len(List_B)
    print('    数据集长度：'+str(Data.shape[0]))
    print('    第一列节点长度：('+str(length_A)+')')
    print('    第二列节点长度：('+str(length_B)+')')
    MaxNodeNum =  int(max(max(List_A),max(List_B))+1)
    print('    节点数量为：'+str(MaxNodeNum))

len_data = NetData.shape[0]

print('Data_shaping...')
Data_Shape(NetData)

def common_neighbours(graph, node1, node2):
    common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))

    return len(common_neighbors)

def resource_allocation(graph, node1, node2):
    ra_score = 0
    common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))

    for common_neighbor in common_neighbors:
        degree = graph.degree(common_neighbor)
        if degree > 0:
            ra_score += 1 / degree

    return ra_score

def adamic_adar(graph, node1, node2):
    aa_score = 0
    common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))

    for common_neighbor in common_neighbors:
        degree = graph.degree(common_neighbor)
        if degree > 1:
            aa_score += 1 / math.log(degree)

    return aa_score

def jaccard_similarity(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    union = len(neighbors1.union(neighbors2))

    if union == 0:
        return 0
    else:
        return intersection / union

def preferential_attachment(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))

    attachment = len(neighbors1) * len(neighbors2)
    return attachment

def hub_promote_index(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    degree_x = graph.degree(node1)
    degree_y = graph.degree(node2)
    max_value = max(degree_x,degree_y)
    hpi = intersection/max_value

    return hpi

def hub_depress_index(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    degree_x = graph.degree(node1)
    degree_y = graph.degree(node2)
    min_value = min(degree_x,degree_y)
    hdi = intersection/min_value

    return hdi

# 计算节点之间的相似度分数
#algorithms = input("请输入要使用的算法（RA、AA、PA、JC、CN、HPI、HDI）：")

def calculate_auc(y_true, y_score):
    gt_pred = list(zip(y_true, y_score))
    probs = []
    pos_samples = [x for x in gt_pred if x[0] == 1]
    neg_samples = [x for x in gt_pred if x[0] == 0]

    # 计算正样本大于负样本的概率
    for pos in pos_samples:
        for neg in neg_samples:
            if pos[1] > neg[1]:
                probs.append(1)
            elif pos[1] == neg[1]:
                probs.append(0.5)
            else:
                probs.append(0)
    # auc = roc_auc_score(true_labels, predicted_scores)
    auc = np.mean(probs)
    #print(auc)
    return auc

print('-'*50)
print('正在计算最优参数...')
def link_prediction(params, algorithm):
    edges = []
    with open(edges_file, 'r') as file:
        for line in file:
            edge = line.strip().split()
            edges.append((int(edge[0]), int(edge[1])))

    # 创建图对象并添加边
    G = nx.Graph()
    G.add_edges_from(edges)

    #algorithm = algorithms
    test_ratio, threshold = params
    # test_ratio = eval(input('请输入划分比例：'))
    # threshold = eval(input('请输入阈值：'))

    edges = list(G.edges())
    random.shuffle(edges)
    test_size = int(len(edges) * test_ratio)
    test_edges = edges[:test_size]
    train_edges = edges[test_size:]

    train_graph = nx.Graph()  # 先划分节点，后创建连边，这时使得不是所有节点都有真实连边，用于后续预测
    train_graph.add_edges_from(train_edges)

    non_connected_node_pairs = [pair for pair in itertools.combinations(train_graph.nodes(), 2)
                                if not train_graph.has_edge(*pair)]

    predicted_edges = []
    y_true = []

    # 计算所有节点对的分数
    all_scores = []
    for u, v in non_connected_node_pairs:
        score = 0
        if algorithm == "RA":
            score = resource_allocation(train_graph, u, v)
        if algorithm == "AA":
            score = adamic_adar(train_graph, u, v)
        if algorithm == "PA":
            score = preferential_attachment(train_graph, u, v)
        if algorithm == "JC":
            score = jaccard_similarity(train_graph, u, v)
        if algorithm == "CN":
            score = common_neighbours(train_graph, u, v)
        if algorithm == "HPI":
            score = hub_promote_index(train_graph, u, v)
        if algorithm == "HDI":
            score = hub_depress_index(train_graph, u, v)
        all_scores.append((u, v, score))

    # 根据分数值排序
    sorted_scores = sorted(all_scores, key=lambda x: x[2], reverse=True)

    # 选取前L个认为是预测存在的边
    L = int(threshold * len(all_scores)) # 假设选取前threshold个边#####################################
    for u, v, score in sorted_scores[:L]:
        predicted_edges.append((u, v))
        if (u, v) in test_edges:
            y_true.append(1)
        else:
            y_true.append(0)

    # 打印预测的边和y_true列表
    # print("Predicted Edges:", predicted_edges)
    # print("y_true:", y_true)

    y_score = [score for u, v, score in sorted_scores[:L]]
    y_pred = [1] * len(y_true)

    auc = calculate_auc(y_true, y_score)

    # Calculate precision
    precision = precision_score(y_true, y_pred)
    return -precision,auc, y_true, y_score

algorithms = ["RA", "AA", "PA", "JC", "CN", "HPI", "HDI"]

def optimize_params_all_algorithms(initial_values):
    results = {}
    for algorithm, init_values in zip(algorithms, initial_values):
        def objective(params):
            precision_fu, auc, y_true, y_score = link_prediction(params, algorithm)
            return precision_fu

        space = [(0.1, 0.9), (0.1, 0.9)]
        res = minimize(objective, x0=init_values, bounds=space)

        _, auc, y_true, y_score = link_prediction(res.x, algorithm)
        results[algorithm] = {
            "test_ratio": res.x[0],
            "threshold": res.x[1],
            "precision": -res.fun,
            "auc": auc,
            "y_true": y_true,
            "y_score": y_score
        }

    return results

if __name__ == "__main__":
    # 提供不同算法的初始值
    initial_values = {
        "RA": [0.7, 0.1],
        "AA": [0.6, 0.1],
        "PA": [0.7, 0.1],
        "JC": [0.4, 0.1],
        "CN": [0.6, 0.1],
        "HPI": [0.6, 0.1],
        "HDI": [0.8, 0.1]
    }

    results = optimize_params_all_algorithms(initial_values.values())

    for algorithm, result in results.items():
        print(f"{algorithm}:")
        print("test_ratio:", result["test_ratio"])
        print("threshold:", result["threshold"])
        print("precision:", result["precision"])
        print("auc:", result["auc"])
        print()

    plt.figure()
    for algorithm, result in results.items():
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_score"])
        roc_auc = aucfunc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{algorithm} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('results_1/fig_roc_all_algorithms.png')
    plt.show()

    plt.figure()
    for algorithm, result in results.items():
        precision, recall, _ = precision_recall_curve(result["y_true"], result["y_score"])
        pr_auc = aucfunc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{algorithm} (area = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('results_1/fig_pr_all_algorithms.png')
    plt.show()

#----------------------------------------------------------------------------------------------------
    algorithms = ["RA", "AA", "PA", "JC", "CN", "HPI", "HDI"]

    for algorithm in algorithms:
        test_ratio_space = np.arange(0.1, 1.0, 0.1)
        threshold_space = np.arange(0.1, 1.0, 0.1)

        precision_list = []
        for test_ratio in test_ratio_space:
            for threshold in threshold_space:
                precision, auc, y_true, y_score = link_prediction([test_ratio, threshold], algorithm)
                precision = -precision
                precision_list.append(precision)

        precision_matrix = np.array(precision_list).reshape(len(test_ratio_space), len(threshold_space))

        fig, ax = plt.subplots()
        im = ax.imshow(precision_matrix, cmap='viridis')
        ax.set_xticks(np.arange(len(threshold_space)))
        ax.set_xticklabels([str(value) for value in threshold_space])
        ax.set_yticks(np.arange(len(test_ratio_space)))
        ax.set_yticklabels([str(value) for value in test_ratio_space])
        plt.xlabel('Threshold')
        plt.ylabel('Test Ratio')

        for i in range(len(test_ratio_space)):
            for j in range(len(threshold_space)):
                text = ax.text(j, i, round(precision_matrix[i, j], 2), ha="center", va="center", color="w")

        plt.colorbar(im)
        plt.title(f'Precision with Test Ratio and Threshold ({algorithm})')

        # 添加图例
        plt.legend([algorithm], loc='upper right', title='Algorithm', title_fontsize='12', shadow=True)

        plt.savefig(f'results_1/fig_precision_1_{algorithm.lower()}.png')
        plt.close()

    print('所有热力图保存完成')