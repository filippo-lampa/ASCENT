from collections import namedtuple, deque
import random

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "v", "p", "p_obs"])

    def add(self, obs, v, p, p_obs):
        """Add a new experience to memory."""

        e = self.experience(obs, v, p, p_obs)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

    def __len__(self):
        """Return the current size of internal memory."""

        return len(self.memory)