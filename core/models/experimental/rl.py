import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from core.models.base import BaseModel


class _MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(64, 64)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCriticRegressor(BaseModel):
    name = "rl_ac"
    group = "experimental"

    def __init__(self, hidden=(64, 64), epochs=1800, lr=3e-3,
                 entropy_weight=0.0, seed=42):
        self._hidden = hidden
        self._epochs = epochs
        self._lr = lr
        self._entropy_weight = entropy_weight
        self._seed = seed
        self._actor = None
        self._critic = None
        self._log_std = None
        self._xs = None
        self._ys = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        torch.manual_seed(self._seed)
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        self._xs = StandardScaler().fit(X)
        self._ys = StandardScaler().fit(y)
        Xt = torch.tensor(self._xs.transform(X), dtype=torch.float32)
        yt = torch.tensor(self._ys.transform(y), dtype=torch.float32)

        self._actor = _MLP(X.shape[1], 1, self._hidden)
        self._critic = _MLP(X.shape[1], 1, self._hidden)
        self._log_std = nn.Parameter(torch.log(torch.tensor([0.5])))
        params = (list(self._actor.parameters())
                  + list(self._critic.parameters()) + [self._log_std])
        opt = torch.optim.Adam(params, lr=self._lr)

        for _ in range(self._epochs):
            opt.zero_grad()
            mu = self._actor(Xt)
            std = self._log_std.exp().clamp(1e-3, 5.0)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            reward = -((action - yt) ** 2)
            value = self._critic(Xt)

            advantage = (reward - value).detach()
            logp = dist.log_prob(action)
            actor_loss = -(logp * advantage).mean()
            entropy = dist.entropy().mean()
            critic_loss = ((value - reward.detach()) ** 2).mean()

            loss = (actor_loss + critic_loss
                    - self._entropy_weight * entropy)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Xt = torch.tensor(self._xs.transform(X), dtype=torch.float32)
        with torch.no_grad():
            mu = self._actor(Xt).numpy()
        return self._ys.inverse_transform(mu).ravel()
