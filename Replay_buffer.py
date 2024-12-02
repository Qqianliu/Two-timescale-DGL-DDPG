import numpy as np

# DDPG :
class Replay_buffer(object):
    def __init__(self, max_memory_cap):
        self.storage = []
        self.max_size = max_memory_cap
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        index = np.random.randint(0, len(self.storage), size=batch_size)
        o, a, o_, r = [], [], [], []

        for i in index:
            obs, action,obs_,R = self.storage[i]
            X, Y, U, R = self.storage[i]
            o.append(np.array(obs, copy=False))
            a.append(np.array(action, copy=False))
            o_.append(np.array(obs_, copy=False))
            r.append(np.array(R, copy=False))
            # d.append(np.array(D, copy=False))

        return np.array(o), np.array(a), np.array(o_), np.array(r).reshape(-1, 1),




# MADDPG
class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, agent_idx):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.concatenate(obs_t[:]))
            actions.append(action)
            rewards.append(reward[agent_idx])
            obses_tp1.append(np.concatenate(obs_tp1[:]))
            dones.append(done[agent_idx])
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, agent_idx):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes, agent_idx)

    def collect(self):
        return self.sample(-1)