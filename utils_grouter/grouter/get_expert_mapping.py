import json


def get_mapping(rank, past_expert_mapping, node_expert):
    rank_node_expert = past_expert_mapping[rank]
    need_remove = []
    need_add = []
    for nid, expert in node_expert.items():
        for eid in expert:
            if eid not in rank_node_expert and rank == nid:
                need_add.append(eid)
            elif eid in rank_node_expert and rank != nid:
                need_remove.append(eid)
    assert len(need_add) == len(need_remove)
    mapping = {}
    for add, remove in zip(need_add, need_remove):
        mapping[add] = remove
    return mapping


def calculate_expert_mapping_dict(expert_placement_config, 
                                  num_nodes, 
                                  expert_per_gpu, 
                                  total_num_expert,
                                  past_expert_mapping=None):
    node_expert = {}
    for node_id, experts in expert_placement_config.items():
        node_expert[int(node_id.split('_')[1])] = [expert["expert_id"] for expert in experts]

    if past_expert_mapping == None:
        past_expert_mapping = {}
        for i in range(num_nodes):
            past_expert_mapping[i] = [j for j in range(i * expert_per_gpu, (i + 1) * expert_per_gpu)]

    total_mapping = {}

    for rank in range(num_nodes):
        total_mapping.update(get_mapping(rank, past_expert_mapping, node_expert))

    for i in range(total_num_expert):
        if i not in total_mapping:
            total_mapping[i] = i
    return total_mapping