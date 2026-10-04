import random

class Environment:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        self.resources = []
        while len(self.resources) < 2:
            pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if pos != self.agent_pos and pos not in self.resources:
                self.resources.append(pos)
        self.collected = []
        return self.get_state()

    def get_state(self):
        x, y = self.agent_pos
        state = [x / self.size, y / self.size]

        # find the nearest uncollected resource
        remaining = [r for r in self.resources if r not in self.collected]
        if remaining:
            distances = [abs(r[0] - x) + abs(r[1] - y) for r in remaining]
            nearest = remaining[distances.index(min(distances))]

            dx = (nearest[0] - x) / self.size
            dy = (nearest[1] - y) / self.size

            direction = [0, 0, 0, 0]  # up, down, left, right
            if dx < 0: direction[0] = 1
            elif dx > 0: direction[1] = 1
            if dy < 0: direction[2] = 1
            elif dy > 0: direction[3] = 1

            state.extend([dx, dy] + direction)
        else:
            # no more resources to collect
            state.extend([0, 0, 0, 0, 0, 0])

        return state  # total length = 8

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1  # move up
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1  # move down
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1  # move left
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1  # move right

        reward = 0
        if self.agent_pos in self.resources and self.agent_pos not in self.collected:
            self.collected.append(self.agent_pos[:])
            reward = 1
            print(f"Resource Collected! reward = {len(self.collected)}")

        done = len(self.collected) == len(self.resources)
        return self.get_state(), reward, done
