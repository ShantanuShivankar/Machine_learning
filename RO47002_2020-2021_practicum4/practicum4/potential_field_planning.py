"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 50.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True


def create_environment(xmin=0, xmax=7, ymin=0, ymax=7, min_distance_start_goal=2, n_obstacles=1, obstacle_radius=1, obstacles_min_separation=3, seed=1):
    np.random.seed(seed)
    # Create obstacles and check that they are separated by a minimum distance
    ox = [np.random.uniform(low=xmin + obstacle_radius, high=xmax - obstacle_radius)]
    oy = [np.random.uniform(low=ymin + obstacle_radius, high=ymax - obstacle_radius)]
    
    for i in range(n_obstacles - 1):
        next_obstacle = False
        while True:
            if next_obstacle:
                break

            ox_candidate = np.random.uniform(low=xmin + obstacle_radius, high=xmax - obstacle_radius)
            oy_candidate = np.random.uniform(low=ymin + obstacle_radius, high=ymax - obstacle_radius)

            for i in range(len(ox)):
                distance_obstacles = np.sqrt((ox_candidate - ox[i])**2 + (oy_candidate - oy[i])**2)
                if distance_obstacles > obstacles_min_separation:
                    if i == (len(ox) - 1):
                        ox.append(ox_candidate)
                        oy.append(oy_candidate)
                        next_obstacle = True
                else:
                    break

    # Create starting point and check that it is not overlapping with the obstacles
    feasible_start = False
    while True:               
        sx = np.random.uniform(low=xmin, high=xmax)
        sy = np.random.uniform(low=ymin, high=ymax)

        for i in range(len(ox)):
            distance_start_obstacles = np.sqrt((sx - ox[i])**2 + (sy - oy[i])**2)
            if distance_start_obstacles > obstacles_min_separation:
                if i == (len(ox) - 1):
                    feasible_start = True
            else:
                break

        if feasible_start:
            break

    # Create goal and check that:
    # 1) starting point and goal are separated by a minimum distance
    # 2) goal is not overlapping with the obstacles
    while True:
        start_goal_separated = False
        # Check 1)
        gx = np.random.uniform(low=xmin, high=xmax)
        gy = np.random.uniform(low=ymin, high=ymax)

        distance_start_goal = np.sqrt((gx - sx)**2 + (gy - sy)**2)
        if distance_start_goal > min_distance_start_goal:
            start_goal_separated = True

        # Check 2)
        feasible_goal = False
        for i in range(len(ox)):
            distance_goal_obstacles = np.sqrt((gx - ox[i])**2 + (gy - oy[i])**2)
            if distance_goal_obstacles > obstacles_min_separation:
                if i == (len(ox) - 1):
                    feasible_goal = True
            else:
                break

        if start_goal_separated and feasible_goal:
            break
            
    return sx, sy, gx, gy, ox, oy


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion

def motion_to_action(ax, ay):
    if ax == 1 and ay == 0:
        action = 0
    elif ax == 0 and ay == 1:
        action = 1
    elif ax == -1 and ay == 0:
        action = 2
    elif ax == 0 and ay == -1:
        action = 3
    elif ax == -1 and ay == -1:
        action = 4
    elif ax == -1 and ay == 1:
        action = 5
    elif ax == 1 and ay == -1:
        action = 6
    elif ax == 1 and ay == 1:
        action = 7
        
    return action
    
    

def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False

def collision_detection(rox, roy, collision_radius=1.0):
    collision = False
    d = np.hypot(rox, roy)
    if np.any(d < collision_radius):
        collision = True
        
    return collision

def potential_field_planning(sx, sy, gx, gy, ox, oy, delay=0.01, reso=0.5, rr=3.0, xmin=0, xmax=7, ymin=0, ymax=7, show=True):
    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
 
    if show:
        #draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.xlim(xmin - 0.2, xmax + 0.2)  # fix the axes of the plot
        plt.ylim(ymin - 0.2, ymax + 0.2)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(sx, sy, "*k")
        plt.scatter(ox, oy, s=5000)
        plt.plot(gx, gy, "*m")

    rx, ry = [sx], [sy]
    rox, roy = [sx - np.array(ox)], [sy - np.array(oy)]
    rgx, rgy = [sx - gx], [sy - gy]
    ra = []
    motion = get_motion_model()
    previous_ids = deque()
    
    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        ax, ay = 0, 0
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
                ax = motion[i][0]
                ay = motion[i][1]

        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)
        rox.append(xp - np.array(ox))
        roy.append(yp - np.array(oy))
        rgx.append(xp - gx)
        rgy.append(yp - gy)
        ra.append(motion_to_action(ax, ay))

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break
            
        if collision_detection(rox[-1], roy[-1]):
            print("Collision detected!")
            break
            

        if show:
            plt.plot(xp, yp, ".r")
            plt.pause(delay)

    ra.append(0)
    rox = np.array(rox)
    roy = np.array(roy)
    n_obstacles = rox.shape[1]
    state_inc = np.append(np.array([rgx, rgy]).reshape(2, -1), rox.reshape(n_obstacles, -1), axis=0)
    state = np.append(state_inc, roy.reshape(n_obstacles, -1), axis=0)

    output = {'state': state.T, 'action': np.array(ra).T}

    return output


def classifier_planning(sx, sy, gx, gy, ox, oy, n_obstacles, clf, scaler, delay=0.01, reso=0.5, xmin=0, xmax=7, ymin=0, ymax=7, rr=3.0, show=True):
    d = np.hypot(sx - gx, sy - gy)
    
    # Actions to numbers
    motions = get_motion_model()
    if show:
        # Plot environment
        plt.xlim(xmin - 0.2, xmax + 0.2)  # fix the axes of the plot
        plt.ylim(ymin - 0.2, ymax + 0.2)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(sx, sy, "*k")
        plt.scatter(ox, oy, s=5000)
        plt.plot(gx, gy, "*m")

    # Initialize trajectory data lists
    rx, ry = [sx], [sy]
    rox, roy = [sx - np.array(ox)], [sy - np.array(oy)]
    rgx, rgy = [sx - gx], [sy - gy]
    ra = []
    success = False
    
    # For path planner
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)
    ix2 = round((sx - minx) / reso)
    iy2 = round((sy - miny) / reso)
    previous_ids = deque()
    
    # Generate trajectory
    counter = 0
    while d >= reso and counter < 25:
        counter += 1
        # Get state vector
        rox_i = np.array(rox[-1])
        roy_i = np.array(roy[-1])
        n_obstacles = rox_i.shape[0]
        state_inc = np.append(np.array([rgx[-1], rgy[-1]]).reshape(2, -1), rox_i.reshape(n_obstacles, -1), axis=0)
        state = np.append(state_inc, roy_i.reshape(n_obstacles, -1), axis=0).T
        
        state_scaled = scaler.transform(state.reshape(1, -1))
        
        # Get action
        action = clf.predict(state_scaled)
        motion_clf = motions[int(action)]
        ax = motion_clf[0]
        ay = motion_clf[1]
        inx = rx[-1] + ax * reso
        iny = ry[-1] + ay * reso
        xp = inx
        yp = iny
        d = np.hypot(gx - xp, gy - yp)
        
        # Append trajectory data to lists
        rx.append(xp)
        ry.append(yp)
        rox.append(xp - np.array(ox))
        roy.append(yp - np.array(oy))
        rgx.append(xp - gx)
        rgy.append(yp - gy)
        ra.append(motion_to_action(ax, ay))
        
        # Run potential field for comparison         
        minp = float("inf")
        minix, miniy = -1, -1
        ax2, ay2 = 0, 0
        for i, _ in enumerate(motions):
            inx2 = int(ix2 + motions[i][0])
            iny2 = int(iy2 + motions[i][1])
            if inx2 >= len(pmap) or iny2 >= len(pmap[0]) or inx2 < 0 or iny2 < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx2][iny2]
            if minp > p:
                minp = p
                minix = inx2
                miniy = iny2
                ax2 = motions[i][0]
                ay2 = motions[i][1]

        ix2 = minix
        iy2 = miniy
        xp2 = ix2 * reso + minx
        yp2 = iy2 * reso + miny
        collision = collision_detection(rox[-1], roy[-1])
        
        if collision:
            print("Collision detected!")
            break

        if show:
            # Plot path
            plt.plot(xp2, yp2, ".r")
            plt.plot(xp, yp, ".m")
            plt.pause(delay)
        
    if d < reso:
        success = True
    elif not collision:
        print('Timeout!')

    ra.append(0)
    return success


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    sx = 0.0  # start x position [m]
    sy = 10.0  # start y positon [m]
    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(False)
        plt.axis("equal")

    # path generation
    rx, ry, rox, roy, rgx, rgy = potential_field_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
