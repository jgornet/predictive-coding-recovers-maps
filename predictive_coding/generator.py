import os
import time
import json
import uuid

import torch
from torch.utils.data import IterableDataset, Dataset
from torch import nn
from torch.nn import functional as f

from torchvision.transforms import ToPILImage

import numpy as np
from pathlib import Path
from rich.progress import Progress

import MalmoPython
import malmoutils
from lxml import etree

import networkx as nx
from skimage.morphology import dilation


torch.multiprocessing.set_sharing_strategy('file_system')

malmoutils.fix_print()

class EnvironmentGenerator(IterableDataset):
    def __init__(self, fn, port, batch_size=128, dataset_size=None, steps=50, tic_duration=0.016):
        super().__init__()
        self.tree = etree.parse(fn)
        self.batch_size = batch_size
        self.agent_host = MalmoPython.AgentHost()
        self.dataset_size = dataset_size
        self.current_samples = 0
        self.steps = steps
        self.tic_duration = tic_duration

        # Load environment
        self.env = MalmoPython.MissionSpec(etree.tostring(self.tree), True)

        # Do not record anything
        self.record = MalmoPython.MissionRecordSpec()

        # Initialize client pool
        pool = MalmoPython.ClientPool()
        info = MalmoPython.ClientInfo('localhost', port)
        pool.add(info)
        experiment_id = str(uuid.uuid1())

        # Initialize environment
        self.agent_host.startMission(
            self.env, pool, self.record, 0, experiment_id)

        # Loop until the mission starts
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        world_state = self.await_ws()
        if len(world_state.video_frames) == 0:
            time.sleep(0.1)
            world_state = self.await_ws()
        frame = world_state.video_frames[-1]
        self.HWC = (frame.height, frame.width, frame.channels)

        self.agent_host.sendCommand("tp 8.5 4.0 2.5")
        time.sleep(0.1)
        world_state = self.await_ws()

        self.init_pathfinding()
        self.path = self.generate_path()
        self.start_time = time.time()
        self.best_index = 0

    def init_pathfinding(self):
        # Get the grid
        world_state = self.agent_host.getWorldState()
        grid = [block == "air" for block in json.loads(
            world_state.observations[-1].text)["board"]]
        grid = ~np.array(grid).reshape((66, 41))
        grid = np.flip(grid, axis=1)
        grid = dilation(grid)

        # Build the graph
        G = nx.grid_graph(dim=grid.shape)
        
        H, W = grid.shape

        edges = []
        for n in G.nodes:
            if n[0] > 0 and n[1] > 0 and (~grid[n[1]-1:n[1]+2, n[0]-1:n[0]+2]).all():
                edges += [(n, (n[0] - 1, n[1] - 1))]
                edges += [((n[0] - 1, n[1] - 1), n)]
            if n[0] > 0 and n[1] < H - 1 and (~grid[n[1]-1:n[1]+2, n[0]-1:n[0]+2]).all():
                edges += [(n, (n[0] - 1, n[1] + 1))]
                edges += [((n[0] - 1, n[1] + 1), n)]

        G.add_edges_from(edges)

        blocks = []
        for n in G.nodes:
            j, i = n
            if grid[i, j]:
                blocks += [n]

        G.remove_nodes_from(blocks)
        self.G = G

    def find_path(self, goal):
        world_state = self.await_ws()
        msg = world_state.observations[-1].text
        x_i = float(json.loads(msg)["XPos"])
        z_i = float(json.loads(msg)["ZPos"])

        i_i = z_i + 29.5
        j_i = -x_i + 19.5

        start = (j_i, i_i)

        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        path = nx.astar_path(self.G, start, goal,
                             heuristic=dist, weight="cost")
        path = [(-j+19.5, i-29.5) for j, i in path]
        path = np.array(list(zip(*path)))

        return path

    def follow_path(self, path, alpha=1, beta=0.6):
        world_state = self.await_ws()
        msg = world_state.observations[-1].text
        x = float(json.loads(msg)["XPos"])
        z = float(json.loads(msg)["ZPos"])
        direction = int(json.loads(msg)["Yaw"])

        distance = np.sqrt((path[0, :] - x)**2 + (path[1, :] - z)**2)
        goal_index = np.argmin(distance) + 1

        if goal_index >= len(distance):
            self.path = self.generate_path()
            self.best_index = 0
            self.start_time = time.time()
            
            return None, False

        if goal_index > self.best_index:
            self.start_time = time.time()
            self.best_index = goal_index

        ntics = (time.time() - self.start_time) / self.tic_duration
        if ntics > 240:
            print(time.time() - self.start_time)
            self.agent_host.sendCommand("tp 8.5 4.0 2.5")
            self.agent_host.sendCommand("move 0")
            self.agent_host.sendCommand("turn 0")
            time.sleep(0.1)
            world_state = self.await_ws()
            self.path = self.generate_path()
            self.best_index = 0
            self.start_time = time.time()

            return None, True

        goal = path[:, goal_index]

        def get_angle(x, z):
            return np.arctan2(-x, z) / np.pi * 180

        target_direction = get_angle(goal[0] - x, goal[1] - z)
        angle_diff = np.mod(target_direction, 360) - np.mod(direction, 360)
        while np.abs(angle_diff) > 180:
            if angle_diff > 180:
                angle_diff += -360
            else:
                angle_diff += 360

        ang_vel = alpha * angle_diff / 180
        speed = beta * np.clip((1 - ang_vel) *
                               distance[goal_index] * 0.3, 0, 1)

        velocity = {"speed": speed, "ang_vel": ang_vel}

        return velocity, False

    def generate_path(self):
        world_state = self.await_ws()
        msg = world_state.observations[-1].text
        x_i = float(json.loads(msg)["XPos"])
        z_i = float(json.loads(msg)["ZPos"])
        direction = int(json.loads(msg)["Yaw"])

        i_i = z_i + 29.5
        j_i = -x_i + 19.5

        start = (int(j_i), int(i_i))

        if not start in self.G.nodes:
            self.agent_host.sendCommand("tp 8.5 4.0 2.5")
            time.sleep(0.1)
            world_state = self.await_ws()

            msg = world_state.observations[-1].text
            x_i = float(json.loads(msg)["XPos"])
            z_i = float(json.loads(msg)["ZPos"])
            direction = int(json.loads(msg)["Yaw"])

            i_i = z_i + 29.5
            j_i = -x_i + 19.5

            start = (int(j_i), int(i_i))

        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        while True:
            goal = (np.random.randint(0, 41), np.random.randint(0, 65))
            try:
                path_ji = nx.astar_path(
                    self.G, start, goal, heuristic=dist, weight="cost")
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound) as e:
                continue
            break
        path = [(-j+19.5, i-29.5) for j, i in path_ji]  # (z, x)
        path = np.array(list(zip(*path)))

        return path

    def __iter__(self):
        return self

    def __next__(self):
        # Initialize array with max sequence length
        H, W, C = self.HWC
        L = self.steps
        inputs = np.empty((L, H, W, C), dtype=np.uint8)
        actions = np.empty((L, 2), np.float32)
        state = np.empty((L, 3), dtype=np.float32)

        # Fill batch
        world_state = self.await_ws()
        pixels = world_state.video_frames[-1].pixels

        for idx in range(L):
            def btoa(pixels):
                return np.reshape(np.frombuffer(pixels, dtype=np.uint8), self.HWC)

            # Fill batch
            world_state = self.await_ws()
            pixels = world_state.video_frames[-1].pixels

            inputs[idx] = btoa(pixels).copy()
        
            msg = world_state.observations[-1].text
            x_i = float(json.loads(msg)["XPos"])
            z_i = float(json.loads(msg)["ZPos"])
            direction = int(json.loads(msg)["Yaw"])

            i_i = z_i + 29.5
            j_i = -x_i + 19.5

            start = (int(j_i), int(i_i))

            state[idx] = np.array([x_i, z_i, direction], dtype=np.float32)

            for _ in range(6):
                velocity, stuck = self.follow_path(self.path)
                if not velocity:
                    if stuck:
                        return None
                    
                    # check if position is feasible
                    if not self.G.has_node(start):
                        self.agent_host.sendCommand(f"move 0")
                        self.agent_host.sendCommand(f"turn 0")
                        self.agent_host.sendCommand("tp 8.5 4.0 2.5")
                        time.sleep(0.1)
                        world_state = self.await_ws()
                    self.path = self.generate_path()
                    self.start_time = time.time()
                    self.best_index = 0
                    velocity, stuck = self.follow_path(self.path)

                speed = velocity["speed"]
                ang_vel = velocity["ang_vel"]
                actions[idx] = [speed, ang_vel]

                self.agent_host.sendCommand(f"move {speed}")
                self.agent_host.sendCommand(f"turn {ang_vel}")

                time.sleep(self.tic_duration)

        return (inputs, actions, state)

    def generate_dataset(self, path: Path, size=1000):
        current_size = 0
        with Progress() as progress:
            task = progress.add_task("Building dataset...", total=size)
            while current_size < size:
                batch = self.__next__()
                if batch is None:
                    continue
                inputs, actions, state = batch
                current_path = path / f"{time.time()}"
                os.makedirs(current_path, exist_ok=True)
                for t in range(len(inputs)):
                    image = ToPILImage()(inputs[t])
                    image.save(current_path / f"{t}.png")
                np.savez(current_path / "actions.npz", actions)
                np.savez(current_path / "state.npz", state)

                current_size += self.steps
                progress.update(task, advance=self.steps)

    def await_ws(self, delay=0.001):
        world_state = self.agent_host.peekWorldState()
        while world_state.number_of_observations_since_last_state <= 0:
            time.sleep(delay)
            world_state = self.agent_host.peekWorldState()

        return world_state
