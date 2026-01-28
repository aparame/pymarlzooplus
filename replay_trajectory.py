import json
import argparse
import sys

import os
import pygame

# Constants for Visualization
CELL_SIZE = 40
GRID_COLOR = (200, 200, 200)
BG_COLOR = (255, 255, 255)
SHELF_COLOR = (0, 0, 0)  # Black for shelves
TEXT_COLOR = (0, 0, 0)
AGENT_RADIUS = 12

# Direction mapping if needed, though we can draw arrows
# UP=0, DOWN=1, LEFT=2, RIGHT=3 from RWARE usually, but JSON might have strings
# episode_runner mapped them to NORTH, SOUTH, WEST, EAST

DIR_VECTORS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
    "EAST": (1, 0),
    # Fallbacks if integers are used
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}


def load_trajectory(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}")
        sys.exit(1)


def draw_grid(screen, grid_width, grid_height):
    for x in range(0, grid_width * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, grid_height * CELL_SIZE))
    for y in range(0, grid_height * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (grid_width * CELL_SIZE, y))


def draw_shelves(screen, shelves):
    for shelf in shelves:
        # Check if shelf is being carried
        # This implementation assumes shelves in 'initial_state' are static
        # unless we update their position based on agents carrying them.
        # However, RWARE shelves disappear/move with agent.
        # For simplicity, we might just draw them at their static pos if not carried,
        # or we rely on agent 'carrying' info to draw them under the agent.

        # In this simple viewer, we'll draw all initial shelves.
        # If the JSON tracks shelf updates, we'd use that.
        # The JSON 'initial_state' has shelves.
        # The steps have 'agents' which have 'carrying' (shelf ID).
        # We will handle drawing carried shelves with the agent.
        # We need to know which shelves are NOT carried to draw them at their original spot.
        # A bit complex because we don't track shelf movement history in the "trajectory" list
        # fully explicitly in the provided snippet logic, only agent state.
        # But wait, rware shelves only move when carried.

        # Let's do this:
        # 1. We keep a state of all shelf positions.
        # 2. In the "replay" loop, if an agent is carrying shelf S, we move shelf S to agent's pos.
        #    If agent puts it down (not implemented in snippet explicitly but implied by 'carrying' going to None),
        #    it stays there.
        # actually, the snippet in episode_runner didn't seem to log shelf positions in every step, only agent info.
        # So we have to INFER shelf positions from 'carrying'.

        # Simplified: Draw initial shelves. If an agent is carrying one, we don't draw it at its old pos?
        # Or we just draw the initial shelves as background and assume they don't visually move
        # unless we add complex state tracking.
        # User snippet: `trajectory_data` has `initial_state`.
        # `current_step_record` has `agents` -> `carrying`.

        # Strategy:
        # We will maintain a `shelf_positions` dict {id: [x, y]}.
        # In each step, if agent carries shelf_id, update shelf_positions[shelf_id] = agent_pos.
        # Then draw.
        pass


def render_step(screen, font, metadata, shelves_dict, step_data):
    screen.fill(BG_COLOR)
    h, w = metadata["grid_size"]

    # 1. Draw Cell Backgrounds based on Carrying
    for agent_info in step_data["agents"]:
        ax, ay = agent_info["pos"]

        # User Request: "loading toggles black, unloading toggles white"
        cell_rect = pygame.Rect(ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        action = agent_info.get("action", "NOOP")
        if action == "TOGGLE_UNLOAD":
            pygame.draw.rect(screen, (255, 255, 255), cell_rect)  # White (Unloading)
        elif action == "TOGGLE_LOAD":
            pygame.draw.rect(screen, (0, 0, 0), cell_rect)  # Black (Loading)
        elif agent_info.get("carrying") is not None:
            pygame.draw.rect(screen, (0, 0, 0), cell_rect)  # Black (Carrying)
        else:
            pygame.draw.rect(screen, (255, 255, 255), cell_rect)  # White (Empty)

    # 2. Draw Grid Lines (on top of backgrounds)
    draw_grid(screen, w, h)

    # Collect shelves being unloaded to skip drawing them so the cell appears white
    shelves_to_skip = set()
    for agent_info in step_data["agents"]:
        if agent_info.get("action") == "TOGGLE_UNLOAD" and agent_info.get("carrying") is not None:
            shelves_to_skip.add(str(agent_info["carrying"]))

    # 3. Draw Shelves
    for s_id, pos in shelves_dict.items():
        if str(s_id) in shelves_to_skip:
            continue
            
        sx, sy = pos
        rect = pygame.Rect(
            sx * CELL_SIZE + 1, sy * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2
        )
        pygame.draw.rect(screen, SHELF_COLOR, rect)

        # Shelf ID text
        text_surf = font.render(str(s_id), True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    # 4. Draw Agents
    for agent_info in step_data["agents"]:
        ax, ay = agent_info["pos"]

        color_name = agent_info.get("color", "blue")
        # specific fix for "yellow" which might be hard to read on white
        c = pygame.Color(color_name)

        center = (
            int(ax * CELL_SIZE + CELL_SIZE / 2),
            int(ay * CELL_SIZE + CELL_SIZE / 2),
        )
        pygame.draw.circle(screen, c, center, AGENT_RADIUS)

        # ID
        id_surf = font.render(str(agent_info["id"]), True, (0, 0, 0))
        id_rect = id_surf.get_rect(center=center)
        screen.blit(id_surf, id_rect)

        # Direction indication (little line or triangle)
        direction = agent_info.get("dir", "SOUTH")
        if direction in DIR_VECTORS:
            dx, dy = DIR_VECTORS[direction]
            end_pos = (center[0] + dx * AGENT_RADIUS, center[1] + dy * AGENT_RADIUS)
            pygame.draw.line(screen, (0, 0, 0), center, end_pos, 3)

    # Info Overlay
    info_text = (
        f"Step: {step_data.get('step', '?')} | success: {metadata.get('success', '?')}"
    )
    info_surf = font.render(info_text, True, (0, 0, 0))
    screen.blit(info_surf, (10, 10))


def main():
    parser = argparse.ArgumentParser(description="Replay a saved trajectory JSON.")
    parser.add_argument("file", help="Path to the .json trajectory file")
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Playback speed (frames per step). Default 2.",
    )
    args = parser.parse_args()

    data = load_trajectory(args.file)
    metadata = data["metadata"]
    init_state = data["initial_state"]
    trajectory = data["trajectory"]

    grid_h, grid_w = metadata["grid_size"]

    pygame.init()
    screen_width = grid_w * CELL_SIZE
    screen_height = grid_h * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Replay: {os.path.basename(args.file)}")
    font = pygame.font.SysFont("Arial", 12)
    clock = pygame.time.Clock()

    # Initialize Shelves state {id: [x,y]}
    # We need to simulate the whole history effectively to know where shelves end up.
    # To support seeking backwards, we'd need to store state snapshots.
    # For now, we'll just allow forward/backward stepping by strictly re-calculating or just simplistic
    # "stateless" if the user didn't request robust simulation (which they didn't, just "view").
    # BUT, shelves move. If we assume they are at initial pos, it will look wrong when carried.

    # Let's Pre-compute shelf positions for every step to allow easy seeking.
    all_shelf_states = []  # list of dicts {id: [x,y]}

    current_shelves = {s["id"]: [s["x"], s["y"]] for s in init_state["shelves"]}

    # Add initial state (step 0 concept)
    all_shelf_states.append(current_shelves.copy())

    # Does trajectory include step 0? usually step 1 is first action result.
    # The JSON structure in episode_runner shows:
    # "agents" list in EACH step has "carrying": shelf_id.
    # Logic:
    # If agent carrying S at step T, then S is at Agent's pos at step T.
    # If agent NOT carrying S, S is where it was at T-1.

    for step in trajectory:
        next_shelves = current_shelves.copy()

        # 1. Identify carried shelves
        carried_map = {}  # shelf_id -> agent_pos
        all_carried_ids = set()

        for ag in step["agents"]:
            sid = ag["carrying"]
            if sid is not None:
                carried_map[sid] = ag["pos"]
                all_carried_ids.add(sid)

        # 2. Update their positions
        for sid, pos in carried_map.items():
            next_shelves[sid] = pos

        # 3. For non-carried shelves, they stay where they were (next_shelves already has prev values)
        # Note: This logic assumes only agents move shelves.

        all_shelf_states.append(next_shelves)
        current_shelves = next_shelves  # Propagate

    # Main Loop
    running = True
    paused = False
    step_idx = 0
    max_step = len(trajectory)

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    step_idx = min(step_idx + 1, max_step - 1)
                    paused = True
                elif event.key == pygame.K_LEFT:
                    step_idx = max(step_idx - 1, 0)
                    paused = True
                elif event.key == pygame.K_r:
                    step_idx = 0  # Restart
                elif event.key == pygame.K_a:
                    paused = False
                    args.fps = 2  # Auto-play at 2 FPS

        # Rendering
        if step_idx < len(trajectory):
            # Trajectory is 0-indexed list of steps.
            # all_shelf_states[0] corresponds to "before step 1"?
            # or we should align them.
            # step_data corresponds to trajectory[step_idx]
            # shelf_state should typically correspond to AFTER the step?
            # In loop above: trajectory[0] -> all_shelf_states[1]

            current_step_data = trajectory[step_idx]
            current_shelf_state = all_shelf_states[
                step_idx + 1
            ]  # +1 because 0 is initial

            render_step(screen, font, metadata, current_shelf_state, current_step_data)
        else:
            # End of replay
            screen.fill(BG_COLOR)
            txt = font.render("End of Replay. Press 'R' to restart.", True, TEXT_COLOR)
            screen.blit(txt, (screen_width // 2 - 100, screen_height // 2))

        pygame.display.flip()

        if not paused and step_idx < max_step - 1:
            step_idx += 1
            clock.tick(args.fps)
        else:
            clock.tick(10)  # reduced framerate when paused/done

    pygame.quit()


if __name__ == "__main__":
    main()
