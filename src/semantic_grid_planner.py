from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


Point = Tuple[int, int]


@dataclass(frozen=True)
class SemanticRisk:
    center_xy: Point
    radius: int
    cost: float
    label: str


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_cost_grid(
    width: int,
    height: int,
    risks: Iterable[SemanticRisk],
    obstacles: Iterable[Point] = (),
    base_cost: float = 1.0,
    obstacle_cost: float = 999999.0,
) -> List[List[float]]:
    grid = [[base_cost for _ in range(width)] for _ in range(height)]

    for risk in risks:
        cx, cy = risk.center_xy
        for y in range(height):
            for x in range(width):
                if manhattan((x, y), (cx, cy)) <= risk.radius:
                    grid[y][x] = max(grid[y][x], risk.cost)

    for x, y in obstacles:
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = obstacle_cost

    return grid


def astar(
    cost_grid: List[List[float]],
    start: Point,
    goal: Point,
    obstacle_cost: float = 999999.0,
) -> Optional[List[Point]]:
    height = len(cost_grid)
    width = len(cost_grid[0]) if height else 0
    open_heap: List[Tuple[float, Point]] = [(0.0, start)]
    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {start: 0.0}

    def neighbors(p: Point) -> Iterable[Point]:
        x, y = p
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                if cost_grid[ny][nx] < obstacle_cost:
                    yield nx, ny

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for nxt in neighbors(current):
            nx, ny = nxt
            tentative = g_score[current] + cost_grid[ny][nx]
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + manhattan(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))

    return None


def path_cost(path: Iterable[Point], cost_grid: List[List[float]]) -> float:
    return sum(cost_grid[y][x] for x, y in path)


def render_ascii(
    width: int,
    height: int,
    path: Iterable[Point],
    start: Point,
    goal: Point,
    risks: Iterable[SemanticRisk],
    obstacles: Iterable[Point] = (),
) -> str:
    canvas = [["." for _ in range(width)] for _ in range(height)]

    for risk in risks:
        cx, cy = risk.center_xy
        for y in range(height):
            for x in range(width):
                if manhattan((x, y), (cx, cy)) <= risk.radius:
                    canvas[y][x] = "r"

    for x, y in obstacles:
        if 0 <= x < width and 0 <= y < height:
            canvas[y][x] = "#"

    for x, y in path:
        canvas[y][x] = "*"

    sx, sy = start
    gx, gy = goal
    canvas[sy][sx] = "S"
    canvas[gy][gx] = "G"

    return "\n".join("".join(row) for row in canvas)

