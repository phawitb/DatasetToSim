"""Audit native gripper state/action values without modifying the dataset."""
import json
from pathlib import Path

import numpy as np
from benchmark72 import DATA, DatasetIndex


def audit():
    dataset = DatasetIndex(DATA)
    rows, states, actions, matches = [], [], [], []
    # Rounded values visible in the user's screenshot; match all six channels.
    screenshot = np.array([5.231, -98.505, 92.879, 65.055, 5.495, .812])
    for index in range(72):
        episode = dataset.episode(index)
        state, action = episode.state_deg[:, 5], episode.action_deg[:, 5]
        states.extend(state.tolist())
        actions.extend(action.tolist())
        rows.append(dict(episode=index, initial_state=float(state[0]),
                         initial_action=float(action[0]), state_min=float(state.min()),
                         state_max=float(state.max()), action_min=float(action.min()),
                         action_max=float(action.max()), frames=len(state)))
        differences = np.linalg.norm(episode.state_deg - screenshot, axis=1)
        frame = int(np.argmin(differences))
        matches.append(dict(episode=index, frame=frame, difference=float(differences[frame]),
                            state=episode.state_deg[frame].tolist(),
                            action=episode.action_deg[frame].tolist()))
    initial = np.array([row['initial_state'] for row in rows])
    values, counts = np.unique(initial, return_counts=True)
    summary = dict(initial_state_range=[float(initial.min()), float(initial.max())],
                   initial_state_counts={str(v): int(n) for v, n in zip(values, counts)},
                   initial_action_range=[min(r['initial_action'] for r in rows), max(r['initial_action'] for r in rows)],
                   all_state_range=[min(states), max(states)], all_action_range=[min(actions), max(actions)],
                   exact_zero_states=states.count(0.), exact_zero_actions=actions.count(0.),
                   screenshot_nearest=sorted(matches, key=lambda row: row['difference'])[:3])
    output = Path(__file__).resolve().parents[1] / 'output/system-identification/gripper-native-data-audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(summary=summary, episodes=rows), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    audit()
