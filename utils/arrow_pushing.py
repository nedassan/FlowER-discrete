import numpy as np
import networkx as nx


def get_sources_and_sinks(delta_matrix, pad_value=-30):
    n = delta_matrix.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool))
    valid_pos = delta_matrix != pad_value

    mask &= valid_pos

    src_idx = np.where(mask & (delta_matrix < 0))
    sink_idx = np.where(mask & (delta_matrix > 0))

    src_vals = -delta_matrix[src_idx]
    sink_vals = delta_matrix[sink_idx]

    sources = [((i, j), val) for i, j, val in zip(src_idx[0], src_idx[1], src_vals)]
    sinks = [((i, j), val) for i, j, val in zip(sink_idx[0], sink_idx[1], sink_vals)]

    return sources, sinks

def build_graph(sources, sinks):
    G = nx.DiGraph()
    G.add_node('S')
    G.add_node('T')

    for node, supply in sources:
        G.add_edge('S', node, capacity=supply)
    for node, demand in sinks:
        G.add_edge(node, 'T', capacity=demand)

    for s, s_cap in sources:
        for t, t_cap in sinks:
            if s[0] in t or s[1] in t:
                cap = min(s_cap, t_cap, 2)
                G.add_edge(s, t, capacity=cap)

    return G

def classify_move(src, sink, electron_count):
    s_diag = (src[0] == src[1])
    t_diag = (sink[0] == sink[1])

    if electron_count == 2:
        if s_diag and not t_diag:
            return "LONE_PAIR_TO_BOND"
        if not s_diag and t_diag:
            return "BOND_TO_LONE_PAIR"
        if not s_diag and not t_diag:
            return "BOND_TO_BOND"
        if s_diag and t_diag:
            return "LONE_PAIR_TO_LONE_PAIR"

    if electron_count == 1:
        return "RADICAL_SHIFT"

    raise ValueError("Unknown electron move")

def aggregate_special_cases(moves):
    sink_map = {}
    new_moves = []

    for move in moves:
        if move['electrons'] == 1:
            sink_map.setdefault(move['sink'], []).append(move)
        else:
            new_moves.append(move)

    for sink, move_list in sink_map.items():
        if len(move_list) > 1:
            src_atoms = [m['src'] for m in move_list]
            new_moves.append({
                'type': 'RADICAL_COMBINATION',
                'src': src_atoms,
                'sink': sink,
                'electrons': sum(m['electrons'] for m in move_list)
            })
        else:
            new_moves.append(move_list[0])

    source_map = {}
    for move in new_moves:
        src_key = move['src']
        source_map.setdefault(src_key, []).append(move)

    homolysis_moves = []
    final_moves = []
    for src, move_list in source_map.items():
        one_e_moves = [m for m in move_list if m['electrons'] == 1]
        two_e_moves = [m for m in move_list if m['electrons'] == 2]

        if len(one_e_moves) > 1 and all(not isinstance(s, list) for s in src):
            sinks = [m['sink'] for m in one_e_moves]
            homolysis_moves.append({
                'type': 'HOMOLYSIS',
                'src': src,
                'sink': sinks,
                'electrons': 2
            })
            final_moves.extend(two_e_moves)
        else:
            final_moves.extend(move_list)

    final_moves.extend(homolysis_moves)

    return final_moves

def get_arrow_pushing(delta_matrix, pad_value=-30):
    sources, sinks = get_sources_and_sinks(delta_matrix, pad_value=pad_value)

    G = build_graph(sources, sinks)

    flow_value, flow_dict = nx.maximum_flow(G, 'S', 'T')

    moves_raw = []
    for (s_idx, s_cap) in sources:
        for (t_idx, t_cap) in sinks:
            f = flow_dict.get(s_idx, {}).get(t_idx, 0)
            if f > 0:
                move_type = classify_move(s_idx, t_idx, f)
                moves_raw.append({
                    'type': move_type,
                    'src': s_idx,
                    'sink': t_idx,
                    'electrons': f
                })

    moves_final = aggregate_special_cases(moves_raw)

    return moves_final
