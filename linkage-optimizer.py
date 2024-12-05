# linkage-optimizer
# 2024 Dec 4
# Yuichi Hirose

import pygame
import pygame_gui
import numpy as np
from scipy.optimize import minimize
import sys
from pygame import gfxdraw

# initialize pygame
pygame.init()

# window size and settings
WIDTH, HEIGHT = 800, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Linkage Optimizer")
clock = pygame.time.Clock()

# create GUI manager
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# color settings
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (102, 178, 255)
GREEN = (0, 153, 153)
RED = (255, 170, 170)
BLACK = (0, 0, 0)

# font
font = pygame.font.SysFont('Calibri', 12)
# font = pygame.font.Font(None, 18)

# initial parameters
radii = [50, 50]
bar_length1 = 100
bar_length2 = 100
angular_speeds = [0.02, 0.02]
positions = [np.array([500, 300]), np.array([600, 300])]  # pulley1, pulley2
init_angles = [0.0, 0.0]
angles = init_angles.copy()

# variables to store previous state
previous_values = {
    "positions": [positions[0].copy(), positions[1].copy()],
    "radii": radii.copy(),
    "bar_lengths": [bar_length1, bar_length2],
    "init_angles": init_angles.copy(),
}

# end point path (trajectory)
path = []

# for slider positions
ui_pos = []
for j in range(13):
    row = []
    for i in range(4):
        row.append([50 + 80 * i, 50 + 50 * j])
    ui_pos.append(row)

adjust = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(ui_pos[0][0], (70, 30)),
    text='adjust',
    manager=manager
)

draw = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(ui_pos[0][1], (70, 30)),
    text='draw',
    manager=manager
)

delete = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(ui_pos[0][2], (70, 30)),
    text='delete',
    manager=manager
)

ik = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(ui_pos[0][3], (70, 30)),
    text='ik',
    manager=manager
)

slider_speed = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[1][0], (300, 20)),
    start_value=angular_speeds[0],
    value_range=(-0.5, 0.5),
    manager=manager,
    container=None
)

slider_r1 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[3][0], (300, 20)),
    start_value=radii[0],
    value_range=(10, 100),
    manager=manager,
    container=None
)

slider_r2 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[4][0], (300, 20)),
    start_value=radii[1],
    value_range=(10, 100),
    manager=manager,
    container=None
)

slider_pos1_x = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[5][0], (300, 20)),
    start_value=positions[0][0],
    value_range=(0, WIDTH),
    manager=manager,
    container=None
)

slider_pos2_x = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[6][0], (300, 20)),
    start_value=positions[1][0],
    value_range=(0, WIDTH),
    manager=manager,
    container=None
)

slider_pos1_y = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[7][0], (300, 20)),  # 適切な位置に配置
    start_value=positions[0][1],
    value_range=(0, HEIGHT),  # y座標は画面の高さ内に限定
    manager=manager,
    container=None
)

slider_pos2_y = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[8][0], (300, 20)),  # 適切な位置に配置
    start_value=positions[1][1],
    value_range=(0, HEIGHT),  # y座標は画面の高さ内に限定
    manager=manager,
    container=None
)

slider_bar1 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[9][0], (300, 20)),
    start_value=bar_length1,
    value_range=(50, 200),
    manager=manager,
    container=None
)

slider_bar2 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[10][0], (300, 20)),
    start_value=bar_length2,
    value_range=(50, 200),
    manager=manager,
    container=None
)

slider_angle1 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[11][0], (300, 20)),
    start_value=init_angles[0],
    value_range=(-np.pi, np.pi),
    manager=manager,
    container=None
)

slider_angle2 = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect(ui_pos[12][0], (300, 20)),
    start_value=init_angles[1],
    value_range=(-np.pi, np.pi),
    manager=manager,
    container=None
)

# rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

# solve constraints to calculate positions of base1, base2, endpoint
def solve_constraints(angle1, angle2):
    base1 = positions[0] + rotation_matrix(angle1) @ np.array([radii[0], 0])
    base2 = positions[1] + rotation_matrix(angle2) @ np.array([radii[1], 0])
    d = np.linalg.norm(base2 - base1)
    if d > bar_length1 + bar_length2 or d < abs(bar_length1 - bar_length2):
        return None, None, None
    phi1 = np.arccos((bar_length1**2 + d**2 - bar_length2**2) / (2 * bar_length1 * d))
    alpha = np.arctan2(base2[1] - base1[1], base2[0] - base1[0])
    endpoint = base1 + rotation_matrix(phi1 + alpha) @ np.array([bar_length1, 0])
    return base1, base2, endpoint

# jacobian inverse kinematics to simulate linkage for the cursor point
def update_angles_to_cursor(target_endpoint):
    max_iterations = 100
    tolerance = 1e-3
    for _ in range(max_iterations):
        _, _, current_endpoint = solve_constraints(angles[0], angles[1])
        if current_endpoint is None:
            break
        error = target_endpoint - current_endpoint
        if np.linalg.norm(error) < tolerance:
            break
        delta = 1e-5

        J = np.zeros((2, 2))
        for j in range(2):
            temp_angles = angles.copy()
            temp_angles[j] += delta
            _, _, endpoint_plus = solve_constraints(temp_angles[0], temp_angles[1])
            J[:, j] = (endpoint_plus - current_endpoint) / delta
        
        delta_angles = np.linalg.pinv(J) @ error
        angles[0] += delta_angles[0]
        angles[1] += delta_angles[1]
        
        # print("gear1 angle: ", angles[0] % (np.pi * 2))
        # print("gear2 angle: ", angles[1] % (np.pi * 2))

# for sliders
def draw_text(screen, text, position, font, color=BLACK):
    """指定した位置にテキストを描画する"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# labels for sliders
def ui():
    draw_text(screen, "velocity", (ui_pos[1][0][0], ui_pos[1][0][1] - 15), font) # slider name
    draw_text(screen, f"{slider_speed.value_range[0]}", (ui_pos[1][0][0] - 25,  ui_pos[1][0][1] + 5), font) # min
    draw_text(screen, f"{slider_speed.value_range[1]}", (ui_pos[1][0][0] + 305, ui_pos[1][0][1] + 5), font) # max

    draw_text(screen, "pulley1 radius", (ui_pos[3][0][0], ui_pos[3][0][1] - 15), font)
    draw_text(screen, f"{slider_r1.value_range[0]}", (ui_pos[3][0][0] - 25,  ui_pos[3][0][1] + 5), font)
    draw_text(screen, f"{slider_r1.value_range[1]}", (ui_pos[3][0][0] + 305, ui_pos[3][0][1] + 5), font)

    draw_text(screen, "pulley2 radius", (ui_pos[4][0][0], ui_pos[4][0][1] - 15), font)
    draw_text(screen, f"{slider_r2.value_range[0]}", (ui_pos[4][0][0] - 25,  ui_pos[4][0][1] + 5), font)
    draw_text(screen, f"{slider_r2.value_range[1]}", (ui_pos[4][0][0] + 305, ui_pos[4][0][1] + 5), font)

    draw_text(screen, "pulley1 x position", (ui_pos[5][0][0], ui_pos[5][0][1] - 15), font)
    draw_text(screen, f"{slider_pos1_x.value_range[0]}", (ui_pos[5][0][0] - 25,  ui_pos[5][0][1] + 5), font)
    draw_text(screen, f"{slider_pos1_x.value_range[1]}", (ui_pos[5][0][0] + 305, ui_pos[5][0][1] + 5), font)

    draw_text(screen, "pulley2 x position", (ui_pos[6][0][0], ui_pos[6][0][1] - 15), font)
    draw_text(screen, f"{slider_pos2_x.value_range[0]}", (ui_pos[6][0][0] - 25,  ui_pos[6][0][1] + 5), font)
    draw_text(screen, f"{slider_pos2_x.value_range[1]}", (ui_pos[6][0][0] + 305, ui_pos[6][0][1] + 5), font)

    draw_text(screen, "pulley1 y position", (ui_pos[7][0][0], ui_pos[7][0][1] - 15), font)
    draw_text(screen, f"{slider_pos1_y.value_range[0]}", (ui_pos[7][0][0] - 25,  ui_pos[7][0][1] + 5), font)
    draw_text(screen, f"{slider_pos1_y.value_range[1]}", (ui_pos[7][0][0] + 305, ui_pos[7][0][1] + 5), font)

    draw_text(screen, "pulley2 y position", (ui_pos[8][0][0], ui_pos[8][0][1] - 15), font)
    draw_text(screen, f"{slider_pos2_y.value_range[0]}", (ui_pos[8][0][0] - 25,  ui_pos[8][0][1] + 5), font)
    draw_text(screen, f"{slider_pos2_y.value_range[1]}", (ui_pos[8][0][0] + 305, ui_pos[8][0][1] + 5), font)

    draw_text(screen, "link1 length", (ui_pos[9][0][0], ui_pos[9][0][1] - 15), font)
    draw_text(screen, f"{slider_bar1.value_range[0]}", (ui_pos[9][0][0] - 25,  ui_pos[9][0][1] + 5), font)
    draw_text(screen, f"{slider_bar1.value_range[1]}", (ui_pos[9][0][0] + 305, ui_pos[9][0][1] + 5), font)

    draw_text(screen, "link2 length", (ui_pos[10][0][0], ui_pos[10][0][1] - 15), font)
    draw_text(screen, f"{slider_bar2.value_range[0]}", (ui_pos[10][0][0] - 25,  ui_pos[10][0][1] + 5), font)
    draw_text(screen, f"{slider_bar2.value_range[1]}", (ui_pos[10][0][0] + 305, ui_pos[10][0][1] + 5), font)

    draw_text(screen, "initial angle 1", (ui_pos[11][0][0], ui_pos[11][0][1] - 15), font)
    draw_text(screen, f"{slider_angle1.value_range[0]:.2f}", (ui_pos[11][0][0] - 25,  ui_pos[11][0][1] + 5), font)
    draw_text(screen, f"{slider_angle1.value_range[1]:.2f}", (ui_pos[11][0][0] + 305, ui_pos[11][0][1] + 5), font)

    draw_text(screen, "initial angle 2", (ui_pos[12][0][0], ui_pos[12][0][1] - 15), font)
    draw_text(screen, f"{slider_angle2.value_range[0]:.2f}", (ui_pos[12][0][0] - 25,  ui_pos[12][0][1] + 5), font)
    draw_text(screen, f"{slider_angle2.value_range[1]:.2f}", (ui_pos[12][0][0] + 305, ui_pos[12][0][1] + 5), font)

# objective function for optimization
def objective(params):
    global radii, bar_length1, bar_length2, positions, angles
    radii[0], radii[1], bar_length1, bar_length2 = params[:4]
    positions[0][0], positions[1][0] = params[4], params[5]
    positions[0][1], positions[1][1] = params[6], params[7]
    angles[0], angles[1] = params[8], params[9]
    
    simulated_path = []
    angle1, angle2 = angles[0], angles[1]
    for t in np.linspace(0, 2 * np.pi, 100):
        angle1 += angular_speeds[0]
        angle2 += angular_speeds[1]
        _, _, endpoint = solve_constraints(angle1, angle2)
        if endpoint is not None:
            simulated_path.append(endpoint)
    
    if len(simulated_path) == 0 or len(target_points) == 0:
        return float('inf')
    
    # error calculation
    error = 0
    for target in target_points:
        closest_point = min(simulated_path, key=lambda p: np.linalg.norm(p - target))
        error += np.linalg.norm(closest_point - target)**2
    
    return error / len(target_points)

# update slider values by the current value
def update_sliders():
    slider_r1.set_current_value(radii[0])
    slider_r2.set_current_value(radii[1])
    slider_bar1.set_current_value(bar_length1)
    slider_bar2.set_current_value(bar_length2)
    slider_pos1_x.set_current_value(positions[0][0])
    slider_pos2_x.set_current_value(positions[1][0])
    slider_pos1_y.set_current_value(positions[0][1])
    slider_pos2_y.set_current_value(positions[1][1])
    slider_angle1.set_current_value(angles[0])
    slider_angle2.set_current_value(angles[1])

# function to return a variable vector and constraint for optimization
def reset_params():
    # get the current parameters as initial parameters and create the variable vector (= parameter set)
    initial_params = (
        radii + 
        [bar_length1, bar_length2] + 
        [positions[0][0], positions[1][0], positions[0][1], positions[1][1]] + 
        angles
    )
    # constraint for optimization
    bounds = [
        (radii[0] - 10, radii[0] + 10),
        (radii[1] - 10, radii[1] + 10),
        (bar_length1 - 20, bar_length1 + 20),
        (bar_length2 - 20, bar_length2 + 20),
        (positions[0][0] - 50, positions[0][0] + 50),
        (positions[1][0] - 50, positions[1][0] + 50),
        (positions[0][1] - 50, positions[0][1] + 50),
        (positions[1][1] - 50, positions[1][1] + 50),
        (angles[0] - np.pi, angles[0] + np.pi),
        (angles[1] - np.pi, angles[1] + np.pi)
    ]
    return initial_params, bounds

# main loop
running = True
auto_update = True  # auto update mode for adjusting parameters using sliders
optimization_active = False
drawing_mode = False
target_points = []
optimization_iterations = 0
counter = 0 # counter to check one full loop in fk

while running:
    time_delta = clock.tick(60) / 1000.0
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        
        # process GUI event
        manager.process_events(event)
        
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            # if adjust button is clicked
            if event.ui_element == adjust:
                auto_update = True
                # reset path
                path = []
                # reset angles to the slider values
                angles[0] = slider_angle1.get_current_value()
                angles[1] = slider_angle2.get_current_value()
                # reset counter
                counter = 0
            elif event.ui_element == ik:
                auto_update = False
            elif event.ui_element == draw:
                auto_update = True
                drawing_mode = True
            elif event.ui_element == delete:
                auto_update = True 
                drawing_mode = False
                target_points.clear()

        # add target points by mouse click
        if event.type == pygame.MOUSEBUTTONDOWN and drawing_mode and not optimization_active and auto_update:
            target_points.append(np.array(pygame.mouse.get_pos()))
        
        # start optimization by hitting space key
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and target_points:
            optimization_active = True
            print("Start optimization")

        # In IK, print pulley angles when clicked  
        if event.type == pygame.MOUSEBUTTONDOWN and not auto_update:
            print("pulley1 angle: ", angles[0] % (np.pi * 2))
            print("pulley2 angle: ", angles[1] % (np.pi * 2))

    # update the current values by slider values
    radii[0] = slider_r1.get_current_value()
    radii[1] = slider_r2.get_current_value()
    bar_length1 = slider_bar1.get_current_value()
    bar_length2 = slider_bar2.get_current_value()
    positions[0][0] = slider_pos1_x.get_current_value()
    positions[1][0] = slider_pos2_x.get_current_value()
    positions[0][1] = slider_pos1_y.get_current_value()
    positions[1][1] = slider_pos2_y.get_current_value()
    speed_value = slider_speed.get_current_value()
    angular_speeds[0] = speed_value
    angular_speeds[1] = speed_value
    init_angles[0] = slider_angle1.get_current_value()
    init_angles[1] = slider_angle2.get_current_value()

    # reset the path when slider values are changed
    if (
        not np.array_equal(positions[0], previous_values["positions"][0]) or
        not np.array_equal(positions[1], previous_values["positions"][1]) or
        not np.array_equal(radii, previous_values["radii"]) or
        bar_length1 != previous_values["bar_lengths"][0] or
        bar_length2 != previous_values["bar_lengths"][1] or
        init_angles[0] != previous_values["init_angles"][0] or
        init_angles[1] != previous_values["init_angles"][1]
    ):
        # reset path
        path = []
        
        # reset counter
        counter = 0

        # store current values
        previous_values["positions"] = [positions[0].copy(), positions[1].copy()]
        previous_values["radii"] = radii.copy()
        previous_values["bar_lengths"] = [bar_length1, bar_length2]
        # previous_values["init_angles"] = init_angles.copy()  # 更新

    if auto_update:
        # reset when initial angles are changed
        if (
            init_angles[0] != previous_values["init_angles"][0]
            or init_angles[1] != previous_values["init_angles"][1]
        ):
            angles[0] = init_angles[0]  # reflect the initial angle
            angles[1] = init_angles[1]  # reflect the initial angle
            previous_values["init_angles"] = init_angles.copy()

        # keep rotating pulleys by delta and store the path
        angles[0] += angular_speeds[0]
        angles[1] += angular_speeds[1]
        counter += np.min(angular_speeds)

        # print("init_angles[0] - angles[0]", init_angles[0] - angles[0])
        base1, base2, endpoint = solve_constraints(angles[0], angles[1])
        
        # if endpoint is not None:
        if endpoint is not None and abs(counter) < (2 * np.pi + 1):
            path.append(endpoint)
    else:
        # ik mode
        if path:
            cursor_pos = np.array(pygame.mouse.get_pos())
            closest_point_ik = min(path, key=lambda p: np.linalg.norm(p - cursor_pos))
            update_angles_to_cursor(closest_point_ik)
    
    # update in optimization mode
    if optimization_active and optimization_iterations < 100:
        if optimization_iterations == 0:
            initial_params, bounds = reset_params()

        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        if result.success:
            radii[0], radii[1], bar_length1, bar_length2 = result.x[:4]
            positions[0][0], positions[1][0] = result.x[4], result.x[5]
            positions[0][1], positions[1][1] = result.x[6], result.x[7]
            angles[0], angles[1] = result.x[8], result.x[9]
            initial_params = result.x  # update initial parameters

            # update slider values
            update_sliders()
            print("Optimization successful. Iteration:", optimization_iterations)
            print("Optimization result: ", result.x)

            # # break when optimization is successful
            # optimization_active = False
            # auto_update = True
        else:
            print("Optimization failed. Iteration:", optimization_iterations)
        optimization_iterations += 1
    elif optimization_iterations >= 100:
        optimization_active = False
        auto_update = True
        optimization_iterations = 0
        print("Finish optimization")
        # reset counter
        counter = 0

    # forward kinematics
    base1, base2, endpoint = solve_constraints(angles[0], angles[1])

    # draw pulleys
    pygame.draw.aacircle(screen, GRAY, positions[0].astype(int), int(radii[0]), 2)
    pygame.draw.aacircle(screen, GRAY, positions[1].astype(int), int(radii[1]), 2)
    
    if base1 is not None and base2 is not None and endpoint is not None:
        # draw links
        pygame.draw.aaline(screen, BLACK, base1.astype(int), endpoint.astype(int), 2)
        pygame.draw.aaline(screen, BLACK, base2.astype(int), endpoint.astype(int), 2)
        
        # # draw base points
        # pygame.draw.circle(screen, RED, base1.astype(int), 5)
        # pygame.draw.circle(screen, RED, base2.astype(int), 5) 

        pygame.draw.aaline(screen, BLACK, positions[0].astype(int), base1.astype(int), 2)
        pygame.draw.aaline(screen, BLACK, positions[1].astype(int), base2.astype(int), 2)

    # draw path
    if len(path) > 1:
        pygame.draw.aalines(screen, RED, False, [p.astype(int) for p in path], 2)

    # draw target points
    for point in target_points:
        pygame.draw.aacircle(screen, GREEN, point.astype(int), 5)

    # draw slider labels
    ui()

    # update and draw sliders
    manager.update(time_delta)
    manager.draw_ui(screen)
    
    pygame.display.flip()
