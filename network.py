#! /home/hong/anaconda3/bin/python

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import operator
from collections import defaultdict
import pandas as pd
import matplotlib.colors as mcolors
#import community as community_louvain

def remove_edge(G, threshold):
    long_edges = list(filter(lambda e: e[2] > threshold, (e for e in G.edges.data('correlation'))))
    le_ids = list(e[:2] for e in long_edges)
    # remove filtered edges from graph G
    G.remove_edges_from(le_ids)
    return G   

def wgcna2nx(f, unwanted, threshold):
    wgcna_data = pd.read_csv(f, sep='\t')
    tmp = pd.DataFrame(np.sort(wgcna_data[['#node1','node2']].values,1))
    tmp['correlation']=wgcna_data.correlation
    nx_data = tmp.drop_duplicates(subset=[0,1])
    network = nx.from_pandas_edgelist(nx_data,0,1,edge_attr='correlation')
    try:
        network.remove_node(unwanted)
    except Exception:
        pass
    network = remove_edge(network, threshold)
    return network 

def degree_analysis(G, name):
    degree_sequence = sorted([d for n, d in G.degree(weight='combined_score')], reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("{}".format(name), figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of {}".format(name))
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.savefig('figures/{}_degree.pdf'.format(name))

def remove_low_degree(G,dthres):
    low_degree = [n for n, d in G.degree() if d < dthres]
    return G.remove_nodes_from(low_degree)

def betweeness_analysis(G,name):
    # largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    # compute centrality
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

    # compute community structure
    lpc = nx.community.label_propagation_communities(H)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}

    #### draw graph ####
    fig, ax = plt.subplots(figsize=(20, 15))
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    node_color = [community_index[n] for n in H]
    node_size = [v * 20000 for v in centrality.values()]
    largest = max(centrality, key=centrality.get)
    second = {i:centrality[i] for i in centrality if i!=largest}
    second_largest = max(second, key=second.get)
    x1, y1 = pos[largest]
    x2, y2 = pos[second_largest]
    plt.text(x1, y1, largest)
    plt.text(x2, y2, second_largest)
#     if name == 'HFHS':
#         third = {i:second[i] for i in second if i!=second_largest}
#         third_largest = max(third, key=third.get)
#         x3, y3 = pos[third_largest]
#         plt.text(x3, y3, third_largest)
    nx.draw_networkx(
        H,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title("Metabolic network for DEG in {}".format(name), font)
    # Change font color for legend
    font["color"] = "r"

    ax.text(
        0.80,
        0.10,
        "node color = community structure",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.06,
        "node size = betweeness centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig('figures/{}_betweeness_annotate.pdf'.format(name))

def bubble(G,name,degre_threshold, color=None, community_colored=False, text_annot=False, gene_list=False, specify_color=False, size_dict=None):
    # largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)
    
    ## full network
    # H = G

    # compute centrality
    centrality = dict(H.degree(weight='combined_score'))

    #### draw graph ####
    
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    if size_dict is None:
        node_size = np.array([v for v in centrality.values()])
        node_size *= 5000.0/node_size.max()
    else:
        node_size = []
        for node in H:
            try:
                node_size.append(size_dict[node]*10)
            except:
                node_size.append(0)
    if community_colored == True:
        community_index = community(H, weight='combined_score')
        community_data = pd.DataFrame.from_dict(community_index, orient='index')
        community_data.to_csv(f'{name}_community.tsv', sep='\t', header=False, index=True)
        node_color = [community_index[n] for n in H]
    if specify_color:
        node_color = []
        for node in H:
            node_color.append(specify_color[node])
    else:
        node_color = np.where(np.fromiter(centrality.values(), dtype=float) > degre_threshold, color, mcolors.TABLEAU_COLORS['tab:gray'])
    #node_color = np.where(np.isin(np.array([a for a in H.nodes()]), np.array(gene_list)), color, mcolors.TABLEAU_COLORS['tab:gray'])
    if not gene_list:
        genes = sorted(centrality, key=centrality.get, reverse=True)
    else:
        genes = gene_list
    draw_params(H, pos, node_color, node_size, genes, color, name, text_annot=text_annot)

def community(H, weight):
    lpc = nx.community.louvain_communities(H, weight)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}
    return community_index

    
def draw_params(H, pos, node_color, node_size, dict_sorted, color, name, text_annot):
    fig, ax = plt.subplots(figsize=(20, 15))
    nx.draw_networkx(
        H,
        pos=pos,
        cmap=plt.get_cmap('coolwarm'),
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.8,
        vmin=-8,
        vmax=8
    )
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin = -8, vmax=8))
    sm._A = []
    plt.colorbar(sm, ax=ax)
    if text_annot:
        for i in dict_sorted:
            x, y = pos[i]
            plt.text(x, y, i, fontsize=24, color=color)
    # x, y = pos[gene_list]
    # plt.text(x, y, gene_list, fontsize=18, color='r')

    # Title/legend
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    #ax.set_title("Metabolic network for DEG in {}".format(name), font)
    # Change font color for legend
    font["color"] = color

    ax.text(
        0.80,
        0.06,
        "node size = perturbed target gene number",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.savefig('figures/{}_bubble_network.pdf'.format(name))
