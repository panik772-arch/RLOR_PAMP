import numpy as np
import torch

from .nets.attention_model.attention_model import *


class Problem:
    def __init__(self, name):
        self.NAME = name


class Backbone(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        problem_name="tsp",
        n_encode_layers=3,
        tanh_clipping=10.0,
        n_heads=8,
        device="cpu",
        k = None
    ):
        super(Backbone, self).__init__()
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})
        self.k = k

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )

        self.decoder = Decoder(
            embedding_dim, self.embedding.context_dim, n_heads, self.problem, tanh_clipping
        )

    def forward(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME,k = self.k)
        input = state.states["observations"]
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)

        # decoding
        cached_embeddings = self.decoder._precompute(encoded_inputs)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)

        return logits, glimpse

    def encode(self, obs):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding)
        cached_embeddings = self.decoder._precompute(encoded_inputs)
        return cached_embeddings

    def decode(self, obs, cached_embeddings):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME, k = self.k)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)

        return logits, glimpse


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self, x):
        logits = x[0]  # .squeeze(1) # not needed for pomo
        return logits


class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Critic, self).__init__()
        hidden_size = kwargs["hidden_size"]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out = self.mlp(x[1])  # B x T x h_dim --mlp--> B x T X 1
        return out


class Agent(nn.Module):
    def __init__(self, embedding_dim=128, device="cpu", name="tsp", k = None):
        super().__init__()
        self.backbone = Backbone(embedding_dim=embedding_dim, device=device, problem_name=name, k = k)
        self.critic = Critic(hidden_size=embedding_dim)
        self.actor = Actor()

    def forward(self, x):  # only actor
        x = self.backbone(x)
        logits = self.actor(x)
        action = logits.max(2)[1]
        return action, logits

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.backbone(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value_cached(self, x, state):
        x = self.backbone.decode(x, state)
        return self.critic(x)

    def get_action_and_value_cached(self, x, action=None, state=None):
        if state is None:
            state = self.backbone.encode(x)
            x = self.backbone.decode(x, state)
        else:
            x = self.backbone.decode(x, state)
        logits = self.actor(x) # as same as mask shape.. 1024,50,51 for example
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample() # (batch, n_traj) e.g. in cvrp_env return action for each batch and trajectory -> (1024,50)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), state


class stateWrapper:
    """
    from dict of numpy arrays to an object that supplies function and data
    """

    def __init__(self, states, device, problem="tsp", k = None):
        self.device = device
        self.states = {k: torch.tensor(v, device=self.device) for k, v in states.items()}
        self.problem = problem
        self.k = k
        if problem == "tsp":
            self.is_initial_action = self.states["is_initial_action"].to(torch.bool)
            self.first_a = self.states["first_node_idx"]
        elif problem == "cvrp":
            input = {
                "loc": self.states["observations"], # (1024,50,2) here 50 are nodes..and depot is 1 node.. so not trajectories, I guess
                "depot": self.states["depot"].squeeze(-1), #(1024,2)
                "demand": self.states["demand"], # (1024,50)
            }
            self.states["observations"] = input
            # action mask (batch, n_traj, all_nodes); current_load (batch, n_traj); demand (batch, customers); depot (batch, xy); last_node (batch, n_traj); observ (batch, customers, xy)
            self.VEHICLE_CAPACITY = 0
            self.used_capacity = -self.states["current_load"]
        elif problem == "cvrp_fleet":
            input = {
                "loc": self.states["observations"],
                "depot": self.states["depot"].squeeze(-1),
                "demand": self.states["demand"],
                "num_veh": self.states["num_veh"]
            }
            self.states["observations"] = input
            self.VEHICLE_CAPACITY = 0
            self.used_capacity = -self.states["current_load"]

    ## Implement the def return_nodes_neighborhood(current_node) and def calculate_the_distance_to_all_nodes(current_node) in the attention_model_wrapper-> stateWrapper

    def return_topK_closest(self):

        '''
        DAR method proposed by Wang2024: Distance-aware Attention Reshaping: Enhance Generalization of Neural Solver
            for Large-scale Vehicle Routing Problems.
        In the future I want to add additional weights, for example based on loads to scale the attention in the decoder not only based on distance, but on anoter factors.
        Consider A-Star Heuristics, for example!
        Or implement the TW VRP! It will be also possible, to attend to time windows!

        Args:
            k: k neighbors for the attention scaling.

        Returns:
            scaled distance matrix! transformed_distances will be then utilized in multi_head attention to scale the u-factor.

        '''
        k = self.k
        if k is None:
            k = int(len(self.states["observations"]["loc"][0]) / 2)

        dist_matrix = self.calculate_the_distance_to_all_nodes()
        #batch_size, num_selected_nodes, num_all_nodes = dist_matrix.shape

        # Step 1: Calculate the distance ranks within each selected node's all_node comparisons. Calculated the rank (0, or 1 is the closest node and the len(nodes)-1 is the largest rank in original array.
        distances_sorted, indices_sorted = torch.sort(dist_matrix, dim=-1)
        ranks = torch.argsort(indices_sorted, dim=-1)

        # Step 2: Apply the transformation across the entire dist_matrix first
        epsilon = 1e-6
        transformed_distances = -torch.log(dist_matrix + epsilon)

        # Step 3: Identify distances not in the top-k and revert their transformation
        not_topk_mask = ranks >= k  # True for distances not in the top-k
        transformed_distances = torch.where(not_topk_mask, -dist_matrix, transformed_distances)

        return dist_matrix, transformed_distances

    def calculate_the_distance_to_all_nodes(self):
        current_node = self.get_current_node() # index
        all_nodes = self.get_observations_with_depot()

        # all_nodes is  tensor of shape (1000, 51, 2)
        # and current_node contains indices of shape (1000, 35)
        # First, create a tensor of batch indices since you're indexing along the second dimension
        batch_indices = torch.arange(all_nodes.size(0)).unsqueeze(1).expand_as(current_node)

        # Use advanced indexing to gather nodes based on indices in current_node
        current_node_coords = all_nodes[batch_indices, current_node]

        all_nodes_exp = all_nodes.unsqueeze(1).expand(-1, current_node.size(1), -1, -1)
        current_node_coords_exp = current_node_coords.unsqueeze(2).expand(-1, -1, all_nodes.size(1), -1)

        # Compute squared distances
        squared_diff = (all_nodes_exp - current_node_coords_exp) ** 2
        dist_matrix = torch.sqrt(squared_diff.sum(dim=-1))

        return dist_matrix


    def get_current_node(self):
        return self.states["last_node_idx"]

    def get_num_veh(self):
        return self.states["num_veh"]

    def get_observations_with_depot(self):
        locs = self.states["observations"]["loc"]
        depot = self.states["observations"]["depot"]
        all_nodes = torch.cat((locs, depot.unsqueeze(1)), dim=1)

        return all_nodes

    def get_mask(self):
        if self.problem == "cvrp_fleet":
            return (1-self.states["action_mask"]).to(torch.bool)
        else:
            return (1-self.states["action_mask"]).to(torch.bool)
