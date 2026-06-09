import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
xml_path   = 'free_floating_2DFK.xml'   # same folder as this script
simend     = 20.0                        # simulation duration [s]
print_camera_config = 0
ee_site_id = -1
# ─────────────────────────────────────────────
#  DESIRED End-Effector position (ee_x, ee_y)
#  Change these to any target configuration you want
# ─────────────────────────────────────────────

# x_desired = np.array([
#     1.20,
#     0.30
# ])

x_desired = np.array([
    0.6,
    0.3
])

current_tau = np.zeros(3)
current_force = np.zeros(2)
# ─────────────────────────────────────────────
#  PID GAINS  (one row per joint: [Kp, Ki, Kd])
#  Tune these to get the response you want.
#  Start conservative — high Kd prevents oscillation
#  in zero-gravity
# ─────────────────────────────────────────────


 # ==================================================
 # Cartesian PD Gains
 # ==================================================

Kp_cart = np.array([
        [20.0, 0.0],
        [0.0, 20.0]
    ])

Kd_cart = np.array([
        [8.0, 0.0],
        [0.0, 8.0]
    ])

# Actuator output saturation  [Nm]
TORQUE_LIMIT = 5.0

# ─────────────────────────────────────────────
#  PID STATE  (persistent across controller calls)
# ─────────────────────────────────────────────
integral_error  = np.zeros(3)
prev_error      = np.zeros(3)
prev_time       = np.zeros(1)   # mutable scalar via array

# Joint address indices — resolved in init_controller() via MuJoCo API
# so we NEVER rely on hardcoded positional guesses
jnt_qposadr = np.zeros(3, dtype=int)   # index into data.qpos  for joint angle
jnt_dofadr  = np.zeros(3, dtype=int)   # index into data.qvel  for joint velocity
act_id      = np.zeros(3, dtype=int)   # index into data.ctrl  for actuator

# ─────────────────────────────────────────────
#  MOUSE / KEYBOARD STATE
# ─────────────────────────────────────────────
button_left   = False
button_middle = False
button_right  = False
lastx = 0
lasty = 0

# ═════════════════════════════════════════════
#  CONTROLLER INIT
# ═════════════════════════════════════════════
def init_controller(model, data):
    """Called once before the sim loop — resolves all joint/actuator indices by name."""
    global integral_error, prev_error, prev_time
    global jnt_qposadr, jnt_dofadr, act_id
    global ee_site_id

    ee_site_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_SITE,
        "ee_site"
    )

    if ee_site_id == -1:
        raise RuntimeError(
            "Site 'ee_site' not found."
        )

    integral_error[:] = 0.0
    prev_error[:]     = 0.0
    prev_time[0]      = data.time

    # ── Resolve joint addresses by NAME (safe, XML-order independent) ──
    joint_names    = ["joint1", "joint2", "joint3"]
    actuator_names = ["joint1", "joint2", "joint3"]  # motor names match joint names in XML

    for i, name in enumerate(joint_names):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"Joint '{name}' not found in model!")
        jnt_qposadr[i] = model.jnt_qposadr[jid]   # where this joint lives in qpos
        jnt_dofadr[i]  = model.jnt_dofadr[jid]    # where this joint lives in qvel

    for i, name in enumerate(actuator_names):
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            # MuJoCo auto-names motors as "joint<n>" if no name attr in XML
            # Try fallback: actuator index = i (motors appear in XML order)
            aid = i
        act_id[i] = aid

    print("=" * 55)
    print("  Free-Floating SMS  —  PID Joint Controller")
    print("=" * 55)
    print(f"  Target EE Position : {x_desired}")
    print(f"  Kp_cart =\n{Kp_cart}")
    print(f"  Kd_cart =\n{Kd_cart}")
    print(f"  Torque limit  : ±{TORQUE_LIMIT} Nm")
    print(f"  qpos addresses : joint1={jnt_qposadr[0]}  joint2={jnt_qposadr[1]}  joint3={jnt_qposadr[2]}")
    print(f"  qvel addresses : joint1={jnt_dofadr[0]}   joint2={jnt_dofadr[1]}   joint3={jnt_dofadr[2]}")
    print(f"  ctrl indices   : {act_id}")
    print("=" * 55)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    site_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_SITE,
        "ee_site"
    )

    mj.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id
    )

    print("\nJacobian Shape =", jacp.shape)
    print(jacp)

# ═════════════════════════════════════════════
#  PID CONTROLLER  (called every step by MuJoCo)
# ═════════════════════════════════════════════
def controller(model, data):

    global prev_time
    global current_tau
    global current_force

    dt = data.time - prev_time[0]

    if dt <= 0.0:
        return


    # ==================================================
    # Current End-Effector Position
    # ==================================================

    ee = data.site("ee_site").xpos

    x_current = np.array([
        ee[0],
        ee[1]
    ])

    # ==================================================
    # Cartesian Position Error
    # ==================================================

    pos_error = x_desired - x_current

    # ==================================================
    # Jacobian
    # ==================================================

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    site_id = mj.mj_name2id(
        model,
        mj.mjtObj.mjOBJ_SITE,
        "ee_site"
    )

    mj.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id
    )

    # ==================================================
    # End-Effector Velocity
    # ==================================================

    ee_vel = jacp @ data.qvel

    vel_xy = np.array([
        ee_vel[0],
        ee_vel[1]
    ])


    # ==================================================
    # Cartesian PD Force
    # ==================================================

    F_task = (
        Kp_cart @ pos_error
        -
        Kd_cart @ vel_xy
    )

    current_force[:] = F_task

    # ==================================================
    # Jacobian Transpose Mapping
    # ==================================================

    J_xy = jacp[0:2, :]

    tau_full = J_xy.T @ F_task

    # ==================================================
    # Extract Manipulator Torques
    # ==================================================

    tau = np.array([
        tau_full[jnt_dofadr[0]],
        tau_full[jnt_dofadr[1]],
        tau_full[jnt_dofadr[2]]
    ])

    # ==================================================
    # Torque Saturation
    # ==================================================

    tau = np.clip(
        tau,
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    )

    current_tau[:] = tau

    # ==================================================
    # Apply Torques
    # ==================================================

    for i in range(3):
        data.ctrl[act_id[i]] = tau[i]

    prev_time[0] = data.time

# ═════════════════════════════════════════════
#  GLFW CALLBACKS
# ═════════════════════════════════════════════
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        # Reset PID state on backspace too
        integral_error[:] = 0.0
        prev_error[:]     = 0.0
        prev_time[0]      = data.time
        print("[INFO] Simulation reset.")

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    if not (button_left or button_middle or button_right):
        return
    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05*yoffset, scene, cam)

# ═════════════════════════════════════════════
#  LOAD MODEL
# ═════════════════════════════════════════════
dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)

model = mj.MjModel.from_xml_path(xml_path)
data  = mj.MjData(model)
cam   = mj.MjvCamera()
opt   = mj.MjvOption()


# ─── GLFW window ──────────────────────────────
glfw.init()
window = glfw.create_window(1200, 900, "Free-Floating SMS — PID Controller", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene   = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

glfw.set_key_callback(window,        keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window,     scroll)

# ─── Camera: top-down planar view ─────────────
cam.azimuth   = 90
cam.elevation = -89
cam.distance  = 6
cam.lookat    = np.array([0.0, 0.0, 0.0])

# ─── Set keyframe home pose ───────────────────
key_id   = model.key("home").id
key_qpos = model.key_qpos[key_id]
data.qpos[:] = key_qpos.copy()
mj.mj_forward(model, data)

# ─── Attach controller ────────────────────────
init_controller(model, data)
mj.set_mjcb_control(controller)

# ═════════════════════════════════════════════
#  DATA LOGGING  (printed to console)
# ═════════════════════════════════════════════
log_interval = 1.0   # seconds between console prints
next_log     = 0.0

# ═════════════════════════════════════════════
#  Video Writing
# ═════════════════════════════════════════════
video_writer = imageio.get_writer(
    "PID_FreeFloating_SMS.mp4",
    fps=60
)
# ═════════════════════════════════════════════
#  Save Time histories to plot joint, torque and base history 
# ═════════════════════════════════════════════
time_history = []

ex_history = []
ey_history = []

tau1_history = []
tau2_history = []
tau3_history = []

Fx_history = []
Fy_history = []
current_tau = np.zeros(3)

base_x_history = []
base_y_history = []
base_yaw_history = []

ee_x_history = []
ee_y_history = []
# ═════════════════════════════════════════════
#  SIMULATION LOOP
# ═════════════════════════════════════════════
print("\n[SIM] Running... (close window or wait for simend)\n")

while not glfw.window_should_close(window):
    time_prev = data.time

    # Step at 500 Hz, render at ~60 fps
    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)         # ← full dynamics + controller
        time_history.append(data.time)
        tau1_history.append(current_tau[0])
        tau2_history.append(current_tau[1])
        tau3_history.append(current_tau[2])
        ee = data.site("ee_site").xpos

        ee_current = ee[:2]

        error = x_desired - ee_current  

        ex_history.append(error[0])
        ey_history.append(error[1])

        Fx_history.append(current_force[0])
        Fy_history.append(current_force[1])

        base_x_history.append(
            data.qpos[0]
        )

        base_y_history.append(
            data.qpos[1]
        )

        base_yaw_history.append(
            np.rad2deg(data.qpos[2])
        )
        ## save ee history for plot
        ee = data.site("ee_site").xpos

        ee_x_history.append(ee[0])
        ee_y_history.append(ee[1])
        if data.time >= simend:
            break

    # ── Console log ───────────────────────────
    if data.time >= next_log:

        ee = data.site("ee_site").xpos

        err = x_desired - ee[:2]

        base_xy = data.qpos[0:2]

        print(
            f"t={data.time:6.2f}s | "
            f"EE=({ee[0]:.3f},{ee[1]:.3f}) | "
            f"Err=({err[0]:.3f},{err[1]:.3f}) | "
            f"Base=({base_xy[0]:.3f},{base_xy[1]:.3f})"
        )

        next_log += log_interval
    if data.time >= simend:
        print("\n[SIM] Reached simend. Close window to exit.")
        break

    # ── Render ────────────────────────────────
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if print_camera_config:
        print(f"cam.azimuth={cam.azimuth}; cam.elevation={cam.elevation}; "
              f"cam.distance={cam.distance}; cam.lookat={cam.lookat}")

    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    rgb = np.zeros(
        (viewport.height,
        viewport.width,
         3),
        dtype=np.uint8
    )

    mj.mjr_readPixels(
    rgb,
    None,
    viewport,
    context
    )

    rgb = np.flipud(rgb)

    video_writer.append_data(rgb)
        
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
video_writer.close()

print("Video saved successfully.")

## EE Error plots ##
plt.figure(figsize=(10,6))

plt.plot(
    time_history,
    ex_history,
    label='X Error'
)

plt.plot(
    time_history,
    ey_history,
    label='Y Error'
)

plt.xlabel('Time [s]')
plt.ylabel('Error [m]')

plt.title(
    'Cartesian Tracking Error'
)

plt.grid(True)
plt.legend()

plt.savefig(
    'Cartesian_Error.png',
    dpi=300,
    bbox_inches='tight'
)


## Torque Plots ##
n = min(len(time_history),
        len(tau1_history))
plt.figure(figsize=(10,6))

plt.plot(
    time_history[:n],
    tau1_history[:n],
    label='Joint 1'
)

plt.plot(
    time_history[:n],
    tau2_history[:n],
    label='Joint 2'
)

plt.plot(
    time_history[:n],
    tau3_history[:n],
    label='Joint 3'
)

plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')

plt.title('Control Torque')

plt.grid(True)
plt.legend()

plt.savefig(
    'Control_Torque.png',
    dpi=300,
    bbox_inches='tight'
)


## Base Reaction Plot ##
plt.figure(figsize=(10,6))

plt.plot(
    time_history,
    base_yaw_history
)

plt.xlabel('Time [s]')
plt.ylabel('Base Yaw [deg]')

plt.title('Spacecraft Base Reaction Motion')

plt.grid(True)

plt.savefig(
    'Base_Reaction_Cartesian_control.png',
    dpi=300,
    bbox_inches='tight'
)


## End-effector Trajectory plot ##
plt.figure(figsize=(7,7))

plt.plot(
    ee_x_history,
    ee_y_history,
    linewidth=2
)

plt.scatter(
    ee_x_history[0],
    ee_y_history[0],
    marker='o',
    s=100,
    label='Start'
)

plt.scatter(
    ee_x_history[-1],
    ee_y_history[-1],
    marker='*',
    s=200,
    label='End'
)

plt.scatter(
    x_desired[0],
    x_desired[1],
    marker='x',
    s=250,
    label='Target'
)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.title(
    'End-Effector Trajectory'
)

plt.axis('equal')

plt.grid(True)
plt.legend()

plt.savefig(
    'EE_Trajectory_Cartesian_control.png',
    dpi=300,
    bbox_inches='tight'
)


## Cartesian Force Plots
plt.figure(figsize=(10,6))

plt.plot(
    time_history,
    Fx_history,
    label='Fx'
)

plt.plot(
    time_history,
    Fy_history,
    label='Fy'
)

plt.xlabel('Time [s]')
plt.ylabel('Force')

plt.title(
    'Cartesian Control Force'
)

plt.grid(True)
plt.legend()

plt.savefig(
    'Cartesian_Force.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()