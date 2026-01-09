import numpy as np
import pyglet
from pyglet import shapes

from gym_go import govars, gogame


def draw_circle(x, y, color, radius):
    num_sides = 50
    verts = [x, y]
    colors = list(color)
    for i in range(num_sides + 1):
        verts.append(x + radius * np.cos(i * np.pi * 2 / num_sides))
        verts.append(y + radius * np.sin(i * np.pi * 2 / num_sides))
        colors.extend(color)
    pyglet.graphics.draw(len(verts) // 2, pyglet.gl.GL_TRIANGLE_FAN,
                         ('v2f', verts), ('c3f', colors))


def draw_command_labels(batch, window_width, window_height):
    pyglet.text.Label('Pass (p) | Reset (r) | Exit (e)',
                      font_name='Arial',
                      font_size=11,
                      x=20, y=window_height - 20, anchor_y='top', batch=batch, multiline=True, width=window_width)


def draw_info(batch, window_width, window_height, upper_grid_coord, state):
    elements = []
    turn = gogame.turn(state)
    turn_str = 'B' if turn == govars.BLACK else 'W'
    prev_player_passed = gogame.prev_player_passed(state)
    game_ended = gogame.game_ended(state)
    info_label = "Turn: {}\nPassed: {}\nGame: {}".format(turn_str, prev_player_passed,
                                                         "OVER" if game_ended else "ONGOING")

    elements.append(pyglet.text.Label(info_label, font_name='Arial', font_size=11, x=window_width - 20, y=window_height - 20,
                      anchor_x='right', anchor_y='top', color=(0, 0, 0, 192), batch=batch, width=window_width / 2,
                      align='right', multiline=True))

    # Areas
    black_area, white_area = gogame.areas(state)
    elements.append(pyglet.text.Label("{}B | {}W".format(black_area, white_area), font_name='Arial', font_size=16,
                      x=window_width / 2, y=upper_grid_coord + 80, anchor_x='center', color=(0, 0, 0, 192), batch=batch,
                      width=window_width, align='center'))
    return elements


def draw_title(batch, window_width, window_height):
    return pyglet.text.Label("Go", font_name='Arial', font_size=20, x=window_width / 2, y=window_height - 20,
                      anchor_x='center', anchor_y='top', color=(0, 0, 0, 255), batch=batch, width=window_width / 2,
                      align='center')



def draw_grid(batch, delta, size, lower, upper):
    label_offset = 20
    elements = []

    for i in range(size):
        pos = lower + i * delta

        # Horizontal line
        elements.append(shapes.Line(
            lower, pos, upper, pos,
            thickness=2,
            color=(77, 77, 77),
            batch=batch
        ))

        # Vertical line
        elements.append(shapes.Line(
            pos, lower, pos, upper,
            thickness=2,
            color=(77, 77, 77),
            batch=batch
        ))

        elements.append(pyglet.text.Label(
            str(i),
            font_name="Arial",
            font_size=11,
            x=lower - label_offset,
            y=pos,
            anchor_x="center",
            anchor_y="center",
            batch=batch
        ))

        elements.append(pyglet.text.Label(
            str(i),
            font_name="Arial",
            font_size=11,
            x=pos,
            y=lower - label_offset,
            anchor_x="center",
            anchor_y="center",
            batch=batch
        ))
    return elements




def draw_pieces_(batch, lower_grid_coord, delta, piece_r, size, state):
    for i in range(size):
        for j in range(size):
            # black piece
            if state[0, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.05882352963, 0.180392161, 0.2470588237],
                            piece_r)  # 0 for black

            # white piece
            if state[1, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.9754120272] * 3, piece_r)  # 255 for white


def draw_pieces(batch, lower, delta, radius, size, state):
    elements = []
    for i in range(size):
        for j in range(size):
            x = lower + i * delta
            y = lower + j * delta

            # Black stone
            if state[0, i, j] == 1:
                elements.append(shapes.Circle(
                    x, y, radius,
                    color=(15, 46, 63),
                    batch=batch
                ))

            # White stone
            elif state[1, i, j] == 1:
                elements.append(shapes.Circle(
                    x, y, radius,
                    color=(248, 248, 248),
                    batch=batch
                ))
    return elements
