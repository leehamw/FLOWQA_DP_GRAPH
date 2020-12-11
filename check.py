import os
import re
from os.path import join
import json
import numpy as np
def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile('[0-9]+_test_dp\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    print(names)
    n_data = len(list(filter(match, names)))
    return n_data
if __name__ =='__main__':
    for i in range(1):
        with open(join('DP_test','{0}_test_dp.json'.format(0))) as f:
            pp=json.load(f)
            graph_arr=pp
            node_num = len(graph_arr['nodes'])
            node_arr = []
            assert len(graph_arr['nodes']) == len(graph_arr['edges'])
            edge_arr = [[0 for ei in range(node_num)] for w0 in range(node_num)]
            for node_index, node in enumerate(graph_arr['nodes']):
                tmp_edge = graph_arr['edges'][node_index]

                node_arr.append(node['word'])

                for ed_idx, edge in enumerate(tmp_edge):

                    if (edge == '' or edge == 'SELF'):
                        continue
                    else:
                        edge_arr[node_index][ed_idx] = 1
            new_node_arr=node_arr
            new_edge_arr = edge_arr
            # for ei ,e in enumerate(edge_arr):
            #     if np.sum(e)==0:
            #         continue
            #     else:
            #         new_node_arr.append(node_arr[ei])
            #         new_edge_arr.append(e)
            print(node_num)
            print(len(new_node_arr))
            print(new_node_arr)
            print(len(new_edge_arr))
            print(len(new_edge_arr[0]))
            for e in new_edge_arr:
                num=np.sum(e)
                print(num)
                print(e)

    # print(count_data('DP_test'))



