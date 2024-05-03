"""
Problem specific node embedding for dynamic feature.
"""

import torch.nn as nn


def AutoDynamicEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "tsp": NonDyanmicEmbedding,
        "cvrp": NonDyanmicEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": NonDyanmicEmbedding,
        "op": NonDyanmicEmbedding,
        "cvrp_fleet": NonDyanmicEmbedding,
        "cvrp_fleet_tw": NonDyanmicEmbedding #CVRPTWDyanmicEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


class SDVRPDynamicEmbedding(nn.Module):
    """
    Embedding for dynamic node feature for the split delivery vehicle routing problem.

    It is implemented as a linear projection of the demands left in each node.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : a class that provide ``state.demands_with_depot`` tensor
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, state):
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            state.demands_with_depot[:, None, :].clone()
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class CVRPTWDyanmicEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(CVRPTWDyanmicEmbedding, self).__init__()

        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, state):

        '''
        doesnt work yet. I need to decide, do I want to process all nodes and for each ratio node generate key, value, logitsX128 ?
        Or do i want to supply all 51 nodes to this function andgenerate 3 embeddings. What does itmean? I use as a key the whole problem state, like a graph context, right?
        # Note, in decoder I add these dynamic embeddings to te nodes of the shape [batch, nodes, 128] so it is basically NO trajectories in here!
        This is important! because here I generate trajectories as well
        '''

        #we have n_traj nodes in each state, so I need to find a way how to solve this first. It make also not much sense to add deadlines to nodes embeddings because deadlines depend on the last n_traj nodes, and so are different for each trajectory
        tw = state.calculate_distance_to_all_nodes() / state.v_ms
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            tw.squeeze()
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic



class NonDyanmicEmbedding(nn.Module):
    """
    Embedding for problems that do not have any dynamic node feature.

    It is implemented as simply returning zeros.

    Args:
        embedding_dim: dimension of output
    Inputs: state
        * **state** : not used, just for consistency
    Outputs: glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
        * **glimpse_key_dynamic** : [batch, graph_size, embedding_dim]
        * **glimpse_val_dynamic** : [batch, graph_size, embedding_dim]
        * **logit_key_dynamic** : [batch, graph_size, embedding_dim]

    """

    def __init__(self, embedding_dim):
        super(NonDyanmicEmbedding, self).__init__()

    def forward(self, state):
        return 0, 0, 0
