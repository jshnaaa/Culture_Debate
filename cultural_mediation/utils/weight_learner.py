"""
Cultural Weight Learning Module
Learns country-specific value weights via projected gradient descent
"""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path


def project_to_simplex(w):
    """
    Project vector w onto the simplex {x: sum(x)=1, x>=0}
    Using Euclidean projection algorithm

    Args:
        w: numpy array of shape (n,)
    Returns:
        projected vector satisfying simplex constraints
    """
    n = len(w)
    u = np.sort(w)[::-1]  # Sort in descending order
    cssv = np.cumsum(u)
    rho_indices = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0]

    if len(rho_indices) == 0:
        # Fallback: uniform distribution
        return np.ones(n) / n

    rho = rho_indices[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(w - theta, 0)


class CountryWeightLearner(nn.Module):
    """
    Learnable country weights with simplex constraints
    """
    def __init__(self, num_countries=75, num_dimensions=5, init_weights_path=None):
        super().__init__()
        self.num_countries = num_countries
        self.num_dimensions = num_dimensions

        # Load initial weights
        if init_weights_path is not None:
            with open(init_weights_path, 'r') as f:
                data = json.load(f)
            country_names = sorted(data['weights'].keys())
            init_weights = np.array([data['weights'][c] for c in country_names])
            self.country_names = country_names
        else:
            # Default: uniform weights
            init_weights = np.ones((num_countries, num_dimensions)) / num_dimensions
            self.country_names = [f"country_{i}" for i in range(num_countries)]

        # Convert to learnable parameter
        self.weights = nn.Parameter(
            torch.tensor(init_weights, dtype=torch.float32)
        )

        # Store initial weights for regularization
        self.register_buffer('init_weights', torch.tensor(init_weights, dtype=torch.float32))

    def forward(self):
        """Return current weights (no projection during forward)"""
        return self.weights

    def project_weights(self):
        """Project weights back to simplex after gradient update"""
        with torch.no_grad():
            for i in range(self.num_countries):
                w = self.weights[i].cpu().numpy()
                w_proj = project_to_simplex(w)
                self.weights[i] = torch.tensor(w_proj, dtype=torch.float32)

    def get_country_weight(self, country_name):
        """Get weight vector for a specific country"""
        if country_name not in self.country_names:
            # Return uniform if country not found
            return torch.ones(self.num_dimensions) / self.num_dimensions
        idx = self.country_names.index(country_name)
        return self.weights[idx]

    def save_weights(self, save_path):
        """Save learned weights to JSON"""
        weights_dict = {
            country: self.weights[i].detach().cpu().numpy().tolist()
            for i, country in enumerate(self.country_names)
        }
        output = {
            "_metadata": {
                "description": "Learned cultural value weights",
                "dimensions": ["autonomy", "order_security", "tradition",
                             "care_universalism", "achievement_power"],
                "constraint": "sum(weights) = 1.0, all weights >= 0"
            },
            "weights": weights_dict
        }
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)


class WeightLearningLoss:
    """
    Combined loss function for weight learning
    """
    def __init__(self,
                 lambda_l2=0.01,
                 lambda_entropy=0.05,
                 lambda_consistency=0.02,
                 similar_countries=None):
        """
        Args:
            lambda_l2: L2 regularization weight
            lambda_entropy: Entropy regularization weight (encourage diversity)
            lambda_consistency: Cultural consistency weight (similar countries should have similar weights)
            similar_countries: List of (country_idx1, country_idx2) pairs that are culturally similar
        """
        self.lambda_l2 = lambda_l2
        self.lambda_entropy = lambda_entropy
        self.lambda_consistency = lambda_consistency
        self.similar_countries = similar_countries or []

    def compute_accuracy_loss(self, predictions, gold_labels):
        """
        Compute cross-entropy loss for predictions

        Args:
            predictions: (batch_size,) tensor of predicted labels (0=Yes, 1=No, 2=Neither)
            gold_labels: (batch_size,) tensor of gold labels
        """
        return nn.CrossEntropyLoss()(predictions, gold_labels)

    def compute_regularization(self, weights):
        """
        Compute regularization terms

        Args:
            weights: (num_countries, num_dimensions) tensor
        """
        # L2 regularization
        l2_reg = torch.norm(weights, p=2) ** 2

        # Entropy regularization (encourage diversity within each country's weights)
        # Higher entropy = more balanced weights
        epsilon = 1e-8  # Avoid log(0)
        entropy = -torch.sum(weights * torch.log(weights + epsilon), dim=1).mean()
        entropy_reg = -entropy  # Negative because we want to maximize entropy

        # Cultural consistency (similar countries should have similar weights)
        consistency_reg = 0.0
        if len(self.similar_countries) > 0:
            for i, j in self.similar_countries:
                consistency_reg += torch.norm(weights[i] - weights[j], p=2) ** 2
            consistency_reg /= len(self.similar_countries)

        return (self.lambda_l2 * l2_reg +
                self.lambda_entropy * entropy_reg +
                self.lambda_consistency * consistency_reg)

    def compute_total_loss(self, predictions, gold_labels, weights):
        """
        Compute total loss = accuracy loss + regularization
        """
        acc_loss = self.compute_accuracy_loss(predictions, gold_labels)
        reg_loss = self.compute_regularization(weights)
        return acc_loss + reg_loss, acc_loss, reg_loss


def define_similar_country_pairs(country_names):
    """
    Define pairs of culturally similar countries for consistency regularization
    Based on geographic/cultural regions
    """
    # Cultural regions
    regions = {
        'nordic': ['sweden', 'norway', 'denmark', 'finland', 'iceland'],
        'east_asian': ['china', 'japan', 'south_korea', 'taiwan', 'hong_kong'],
        'middle_eastern': ['egypt', 'saudi_arabia', 'iraq', 'syria', 'lebanon',
                          'iran', 'palestinian_territories'],
        'south_asian': ['india', 'pakistan', 'bangladesh', 'nepal', 'sri_lanka'],
        'southeast_asian': ['thailand', 'vietnam', 'malaysia', 'singapore',
                           'indonesia', 'philippines', 'myanmar', 'cambodia', 'laos'],
        'latin_american': ['brazil', 'argentina', 'chile', 'colombia', 'mexico',
                          'peru', 'venezuela'],
        'anglo': ['united_states_of_america', 'united_kingdom', 'canada',
                 'australia', 'new_zealand'],
        'eastern_european': ['poland', 'hungary', 'romania', 'ukraine', 'russia',
                            'serbia', 'croatia', 'bosnia_and_herzegovina'],
        'sub_saharan_african': ['kenya', 'ethiopia', 'somalia', 'south_africa',
                               'zimbabwe', 'south_sudan', 'sudan']
    }

    similar_pairs = []
    for region_countries in regions.values():
        # All pairs within the same region
        region_indices = []
        for country in region_countries:
            if country in country_names:
                region_indices.append(country_names.index(country))

        # Add all pairs
        for i in range(len(region_indices)):
            for j in range(i + 1, len(region_indices)):
                similar_pairs.append((region_indices[i], region_indices[j]))

    return similar_pairs


# Training function
def train_weights(weight_learner,
                  train_loader,
                  loss_fn,
                  num_epochs=10,
                  lr=0.01,
                  device='cuda'):
    """
    Train country weights using projected gradient descent

    Args:
        weight_learner: CountryWeightLearner instance
        train_loader: DataLoader yielding (country_idx, agent_responses, gold_label)
        loss_fn: WeightLearningLoss instance
        num_epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    weight_learner = weight_learner.to(device)
    optimizer = torch.optim.Adam(weight_learner.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc_loss = 0.0
        total_reg_loss = 0.0

        for batch_idx, (country_idx, agent_responses, gold_label) in enumerate(train_loader):
            # Get country weights
            country_weights = weight_learner.weights[country_idx]  # (batch_size, 5)

            # Compute predictions (this would be done by the multi-agent system)
            # For now, placeholder
            # predictions = multi_agent_predict(agent_responses, country_weights)

            # Compute loss
            # loss, acc_loss, reg_loss = loss_fn.compute_total_loss(
            #     predictions, gold_label, weight_learner.weights
            # )

            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Project weights back to simplex
            weight_learner.project_weights()

            # total_loss += loss.item()
            # total_acc_loss += acc_loss.item()
            # total_reg_loss += reg_loss.item()

        # Print epoch stats
        # avg_loss = total_loss / len(train_loader)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return weight_learner


if __name__ == "__main__":
    # Test simplex projection
    w = np.array([0.5, 0.3, -0.1, 0.4, 0.2])
    w_proj = project_to_simplex(w)
    print(f"Original: {w}")
    print(f"Projected: {w_proj}")
    print(f"Sum: {w_proj.sum()}, All non-negative: {(w_proj >= 0).all()}")

    # Test weight learner
    init_path = Path(__file__).parent.parent / "data" / "country_weights_init.json"
    learner = CountryWeightLearner(init_weights_path=init_path)
    print(f"\nLoaded {learner.num_countries} countries")
    print(f"Egypt weights: {learner.get_country_weight('egypt')}")

    # Test projection
    learner.weights.data += torch.randn_like(learner.weights) * 0.1
    print(f"\nBefore projection - Egypt sum: {learner.weights[learner.country_names.index('egypt')].sum()}")
    learner.project_weights()
    print(f"After projection - Egypt sum: {learner.weights[learner.country_names.index('egypt')].sum()}")
