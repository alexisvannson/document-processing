import math
import statistics


def skew_angle(tokens):
    """Median slope of the words' top edges. Photos are often tilted a few degrees, which is
    enough to put a right-aligned price one line below its label."""
    angles = []
    for t in tokens:
        (x1, y1), (x2, y2), _, (x4, y4) = t["polygon"]
        if abs(x2 - x1) > abs(y4 - y1):  # wider than tall: the top edge is a reliable slope
            angles.append(math.atan2(y2 - y1, x2 - x1))
    return statistics.median(angles) if angles else 0.0


def group_lines(tokens):
    """
    Groups word tokens into text lines, top to bottom, each line left to right.
    Coordinates are first rotated by the page's skew; then a token joins the current line
    when its vertical centre is within half a typical word height of the line's centre.

    tokens: list of dicts with a "polygon" ([[x, y] x 4]); returns lists of token indices.
    """
    if not tokens:
        return []

    angle = skew_angle(tokens)
    cos, sin = math.cos(angle), math.sin(angle)
    xs = [[x * cos + y * sin for x, y in t["polygon"]] for t in tokens]
    ys = [[-x * sin + y * cos for x, y in t["polygon"]] for t in tokens]

    centres = [(min(y) + max(y)) / 2 for y in ys]
    tolerance = statistics.median(max(max(y) - min(y), 1) for y in ys) / 2

    lines = []  # [(centre, [token indices])]
    for i in sorted(range(len(tokens)), key=lambda i: centres[i]):
        if lines and abs(centres[i] - lines[-1][0]) <= tolerance:
            centre, members = lines[-1]
            members.append(i)
            lines[-1] = (sum(centres[j] for j in members) / len(members), members)
        else:
            lines.append((centres[i], [i]))

    return [sorted(members, key=lambda i: min(xs[i])) for _, members in lines]
