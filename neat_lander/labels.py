"""Names for the 8 observations and 4 actions of LunarLander-v2.

neat-python numbers input nodes -1..-8 and output nodes 0..3, so `INPUT_NAMES[i]` belongs to
node -(i + 1) and `ACTION_NAMES[k]` to node k.
"""

INPUT_NAMES = ['x position', 'y position (altitude)', 'horizontal velocity', 'vertical velocity',
               'angle', 'angular velocity', 'left leg contact', 'right leg contact']
INPUT_SHORT = ['x', 'y', 'vx', 'vy', 'angle', 'spin', 'L leg', 'R leg']
ACTION_NAMES = ['do nothing', 'left engine', 'main engine', 'right engine']
ACTION_SHORT = ['none', 'left', 'main', 'right']


def node_name(key, short=True):
    """Human name for a neat-python node key."""
    if key < 0:
        return (INPUT_SHORT if short else INPUT_NAMES)[-key - 1]
    if key < len(ACTION_NAMES):
        return (ACTION_SHORT if short else ACTION_NAMES)[key]
    return f'h{key}'
