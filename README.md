# opensim-frameworks

`opensim-frameworks` is a Python package that extends
[OpenSim](https://opensim.stanford.edu/) with anthropometric models, adjustable
indoor-cycling equipment, model composition, and visualization and export
tools.

## Features

- **`User`**: full-body OpenSim model based on the Rajagopal-Lai-Uhlrich 2023
  model and scaled using ANSUR II anthropometric references. It exposes
  measurements, joint centres, markers, and explicit measurement and posture
  setters.
- **`Bike`**: adjustable indoor-cycling bike model with rider fitting, pedals,
  handlebar, screen and support, and a rounded, tapered saddle.
- **STEP import**: converts STEP geometry into OpenSim bodies and VTP meshes,
  using material densities and colours when available.
- **`Framework`**: named container for separate OpenSim models, with composite
  snapshots, couplings, and tools for simulation or standalone model creation.
- **Visualization**: `VTKViewer` and `VTKVisualizer` display models. The
  interactive player supports `.mot` and `.sto` motions; export tools save SVG
  snapshots and MP4 animations.

## Architecture

The package separates models, composition, and visualization:

```text
src/opensim_frameworks/
├── models/
│   ├── user/       User, ANSUR II references, scaling, and anatomical assets
│   └── equipment/
│       ├── equipment.py    Equipment, Screen, and STEP import
│       └── bikes/
│           └── bike/       Parametric Bike and kinematic fitting
├── bodies/         Geometric primitives and OpenSim body builders
├── framework/      Model composition and couplings
├── viewer/         VTK scene, player, off-screen animation, and SVG export
└── utils/
    ├── opensim.py    OpenSim, frame, and geometry utilities
    ├── step.py       STEP parsing, tessellation, and tree utilities
    └── constants.py Shared package constants
```

`User` and `Bike` are native OpenSim models. `Framework` keeps its members as
separate models and builds a composite snapshot when simulation, coupling, or
visualization is needed. Framework couplings belong to the snapshot and do not
modify the source models. `VTKViewer` is the public facade for models and
frameworks; `VTKVisualizer` manages the VTK scene, and the interactive player
provides playback controls.

## Requirements and installation

The project requires **Python 3.10 or later**. Using the models also requires
Python bindings for OpenSim that are compatible with the selected interpreter.
Check the OpenSim version in the target environment; the package does not
define a Python/OpenSim compatibility matrix.

The distribution name is `opensim-frameworks`; import the Python namespace as
`opensim_frameworks`.

Install the package with the OpenSim extra from a local repository checkout:

```powershell
python -m pip install -e ".[opensim]"
```

To import STEP files, install the `step` extra, which adds `pythonocc-core`:

```powershell
python -m pip install -e ".[opensim,step]"
```

To install the test dependencies as well:

```powershell
python -m pip install -e ".[opensim,step,test]"
```

The `opensim` extra requires bindings that can be installed with `pip` in the
current environment. If OpenSim is managed by Conda or a separate
installation, use that environment and confirm that `import opensim` works
before importing `opensim_frameworks`. On Windows, the package prepares the
current Python environment's native-library paths before loading OpenSim.
`requirements.txt` also includes `pythonocc-core>=7.7` for STEP import.

The OpenSim model, ANSUR II data, and meshes used by `User` are included in the
package. `Bike` uses bundled primitives and meshes.

## Quick start

```python
from pathlib import Path

from opensim_frameworks import Bike, VTKViewer

# Create and fit the bike to the reference rider.
bike = Bike()
framework = bike.fit_user()
user = framework["user"]

print(user.height_cm, user.mass_kg)
print(bike.screen_center)  # World coordinates in mm.

viewer = VTKViewer()
viewer.save_snapshot(framework, path=Path("output") / "bike_user.svg")
```

If no user is passed to `fit_user()`, the method creates a male reference user
with a stature of **170 cm** and a mass of **70 kg**. The path passed to
`save_snapshot()` can be a string or `pathlib.Path`; destination directories
are created automatically.

## Anthropometric `User` model

```python
from pathlib import Path

from opensim_frameworks import User

user = User("F", height_cm=170.0, mass_kg=65.0)
user.set_left_arm_length(user.left_arm_length * 1.05)
user.set_neck_flexion(15.0)
output = Path("output")
output.mkdir(exist_ok=True)
user.to_osim(output / "user.osim")
```

The constructor accepts `gender` (`"M"` or `"F"`), `height_cm` or
`percentile` (default 50), an optional `mass_kg`, and optionally a path to an
alternative ANSUR II dataset. When `height_cm` is supplied, anthropometric
measurements are resolved from stature; otherwise, the requested percentile is
used. If `mass_kg` is omitted, the model's total mass is calibrated to the
weight resolved from ANSUR II. A stature outside the observed range can be
extrapolated and raises a `UserWarning`.

### Anthropometric scaling

With `height_cm`, each measurement is estimated using a per-measurement PCHIP
regression against ANSUR stature. The resulting empirical percentile is
informational and does not replace the requested height. Without `height_cm`,
the common percentile is calculated independently for each numeric
measurement; a single ANSUR subject is not selected. Each segment is scaled
using ANSUR measurements associated with its available axes, with the stature
ratio used for the remaining axes. After geometric scaling, body masses and
inertias are rescaled uniformly so the model's total mass matches `mass_kg` or
the weight resolved from ANSUR data. The pelvis is translated vertically so
the lowest vertices of both scaled foot meshes touch the ground plane (`y = 0`)
without changing joint angles. This uses the displayed foot geometry as an
approximation; it does not simulate soft tissue, contact forces, or footwear.
The foot markers remain at their anatomical offsets above the ground.

### ANSUR-resolved values

The `anthropometry` property exposes the resolved ANSUR reference, including
its `gender`, `percentile`, and `values`. The `values` mapping contains numeric
ANSUR columns in their original dataset units, predominantly millimetres,
with `stature_m` in metres. Its `height_m` and `height_cm` convenience
properties are derived from `stature_m`.

The `height_cm` property is in centimetres and `mass_kg` is in kilograms.
Anthropometric lengths and circumferences exposed by properties, markers, and
model spatial coordinates use OpenSim SI units (metres). Right- and
left-side measurements are independent: for example,
`set_left_arm_length()` changes the left side without scaling the right.

### Changing posture

The cervical joints provide neck-base flexion/extension and lateral flexion
(`neck_base_flexion`, `neck_base_lateral_flexion`), and head flexion/extension,
lateral flexion, and rotation (`neck_flexion`, `neck_lateral_flexion`,
`neck_rotation`). The wrists support flexion/extension and radial/ulnar
deviation. `REYE`/`LEYE` mark the eye sockets, `RPALM`/`LPALM` mark the palm
centres, and foot markers identify the heel, toe, and fifth metatarsal; there
is no first-metatarsal marker in this model. Each foot has a subtalar degree
of freedom for inversion/eversion.

`to_osim()` exports the scaled model in its current pose. By default, it also
copies referenced meshes into a `Geometry` directory next to the `.osim` file.
`show()` opens the interactive viewer for the model.

#### Raw posture setters

Posture setters write the raw OpenSim coordinate value. Call
`update_state()` before reading values derived from the new posture, such as
joint centres, centre of mass, or muscle lengths and forces. Posture getters
read the coordinate directly and do not require `update_state()`; they return
the value actually held by OpenSim, including any coordinate clamping.

### Joint centres and derived measurements

Joint-centre properties return the current child-frame origin in the ground
frame, in metres. Derived geometric measurements are calculated from those
centres at the current posture and are not ANSUR measurements. For example,
`shoulder_width` is the distance between the shoulder joint centres, whereas
`biacromial_breadth` is an ANSUR measurement; they are independent and
generally do not have the same value.

### Known update_state() limitation

The patella coordinates (`knee_angle_l_beta` and `knee_angle_r_beta`) are
coupled to the corresponding knee coordinates by
`CoordinateCouplerConstraint`. After changing a knee angle, a plain
`update_state()` does not resolve that coupling. Use OpenSim assembly to
enforce the constraint before relying on the resulting patella joint-centre
positions.

## Bike and fitting

The bike's world frame uses `x` forward, `y` upward, and `z` to the right.
Lengths and positions in the public `Bike` API are in **millimetres**, angles
in **degrees**, forces in newtons, and densities in kg/m³; OpenSim calculations
remain in SI units. The frame, saddle, mounts, and handlebar are fixed; the
crankset and pedals rotate.

`Bike(...)` configures the seat-tube, handlebar-tube, and slide ranges and
steps; crank-axis position and crank length; pedal and handlebar dimensions;
component densities; and braking force. For example, `seat_tube_steps=20`
defines 21 positions (indices 0–20), while `seat_tube_step` selects one
position. If omitted, the middle index is selected.

To run the fitting procedure:

```python
from opensim_frameworks import Bike, User

bike = Bike()
user = User("M", height_cm=178.0, mass_kg=75.0)
framework = bike.fit_user(
    user,
    shoe_height_mm=30.0,
    trunk_angle_deg=45.0,
    shoulder_flexion_deg=75.0,
    elbow_flexion_deg=15.0,
)
```

`fit_user()` modifies the bike and user, then returns a `Framework` containing
both models and the feet coupled to the pedals. Default posture targets are a
45° trunk angle, 75° shoulder flexion, and 15° elbow flexion. The default user
is the male reference at 170 cm and 70 kg. During inverse kinematics, shoulder
and elbow flexion are shared bilaterally, while shoulder adduction remains
independent on each side. The final fit is available in `bike.fit_report`. If
the selected geometry cannot satisfy both the target height and hand position,
fitting retains the best result and emits a `UserWarning` describing the
residuals.

Tube angles are derived from the start and end points of their ranges rather
than specified directly. `seat_tube_range_mm` and
`handlebar_tube_range_mm` each use `((x0, y0), (x1, y1))` in global
coordinates (millimetres, with z always zero); `seat_tube_steps` and
`handlebar_tube_steps` specify the number of intervals. Seat and handlebar
slides are relative to the top of their tube: `seat_slide_range_mm` and
`handlebar_slide_range_mm` are one-dimensional `(start, end)` ranges in
millimetres, with `*_slide_steps` and `*_slide_angle_deg` defining their
positions and direction in the XY plane. `seat_tube_angle_deg` and
`handlebar_tube_angle_deg` are read-only properties derived from these points.
`frame_tube_diameter_mm` defines the frame-tube diameter.

Directed angles in the XY plane are measured from +X toward +Y by the
right-hand rule (positive rotation about +Z). A seat tube that rises backward
has an angle greater than 90° (about 104° by default). The top of the screen
tilts according to `screen_inclination_deg` (40° by default).
`horn_pitch_deg` raises the horns when positive; positive `horn_roll_deg`
rotates them medially, toward the bike's centre plane. The two horns are
oriented independently; the handlebar has no global tilt. The default
`crank_axis_position_mm` is `(0, 322, 0)`.

Fitting centres the midpoint between the `TOE` and `MT5` markers on the pedal
and sets its distance from the pedal surface, measured along the surface
normal, to `shoe_height_mm`. In pedal-local coordinates, the midpoint is
centred along the surface, and its distance from the pedal axle is half the
pedal thickness plus `shoe_height_mm`. The feet are rigidly coupled to their
pedals, so this distance is maintained as the crank and pedals rotate. The
right foot and pedal remain horizontal; the left pedal follows the foot
inclination while the left ankle remains neutral. Inverse kinematics also
minimizes the alignment error between the sole plane and pedal top, and
between the longitudinal axes of the foot and pedal. Subtalar rotation can
correct sole inclination, but orientation targets remain subordinate to
placing the markers on the pedal. The fit and residuals are available in
`bike.fit_report`.

With `allow_ik_adjustment=False`, the requested trunk, shoulder, and elbow
angles remain fixed while the fitting procedure can adjust other degrees of
freedom. Setting `trunk_angle_deg=None` uses the initial lumbar posture as a
reference but allows inverse kinematics to adjust it to reach the handlebar
horns. In this mode, the handlebar height is selected to approach the saddle
height.

## Visual ergonomics

`VisualErgonomicsFramework`, exported from `opensim_frameworks.framework`,
extends `Framework` with a rectangular visual cone anchored at the midpoint
between the eyes and surfaces representing hand reachability. Each hand is
articulated relative to the radius by a joint supporting wrist flexion/
extension and radial/ulnar deviation. The `RPALM` and `LPALM` markers are
attached to the hand bodies, identify the palm centres, and are positioned
using anthropometric palm length. The cone opens forward; its upper edge is
parallel to the head plane through the eyes and its vertical extent points
downward. The line of sight has a separate orientation. Overlays are included
in the framework and appear in the viewer.

```python
from opensim_frameworks import Bike, User
from opensim_frameworks.framework import VisualErgonomicsFramework

user = User("M", height_cm=178.0, mass_kg=75.0)
bike = Bike()
bike.fit_user(user)
ergonomics = VisualErgonomicsFramework(
    user,
    bike,
    line_of_sight_offset_deg=-15.0,
)

posture_optimized = ergonomics.optimize_user_position(
    neck_base_flexion_deg=(-30.0, 30.0),
    lumbar_framework_deg=0.0,
    right_shoulder_flexion_deg=(20.0, 90.0),
)
optimized = posture_optimized.optimize_screen_position(
    screen_horizontal_offset_mm=(150.0, 250.0),
    screen_vertical_offset_mm=(32.0, 100.0),
    screen_incline_range_deg=(10.0, 90.0),
)
view_state = optimized.get_view_state()
```

`line_of_sight_offset_deg` sets the line-of-sight angle relative to the head
plane through the eyes; the default `-15°` points downward. It is independent
of the cone's size: the upper edge always coincides with the head plane, while
`cone_height_angle_deg` sets the downward extent. `optimize_screen_position`
does not change the `User` posture. It makes the screen perpendicular to the
current line of sight and moves it along that line to the nearest position
satisfying reachability and equipment-position constraints. If the requested
inclination is outside the supplied range, the nearest allowed inclination is
used and a `UserWarning` is emitted before positioning continues. A
`ValueError` may still be raised if the line of sight cannot be represented by
the screen model or if no position satisfies the other constraints.

`optimize_user_position` runs sequential stages to maximize reachability,
restore the palm positions, and align the line of sight with the screen
centre. If arm or neck ranges prevent exact restoration or alignment, the
closest allowed posture is retained and a `UserWarning` is emitted instead of
aborting. Palm positioning uses bounded least squares; gaze optimization
selects the neck coordinates with the most independent effects and optimizes
them within their bounds.

For screen parameters, a number fixes an offset or inclination, while a
two-number tuple defines a range. Unspecified offsets remain free within the
equipment constraints; inclination is determined by perpendicularity to the
line of sight. With `inplace=False`, the method returns an independent
framework; with `inplace=True`, it updates only the current framework's screen
and returns `None`. For `User` coordinates, `None` preserves the current value,
a number fixes it, and a two-number tuple defines an optimization range.
`optimize_user_position` exposes neck, arm, elbow, forearm, wrist, and lumbar
flexion coordinates while keeping the screen fixed. By default, neck-base
flexion and head flexion are optimized from -45° to 45°, and head rotation
from -30° to 30°. If a value or range exceeds a parameter's current limits,
the limits are expanded and a warning is emitted. If joint ranges prevent the
palm-restoration or gaze-alignment target, the admissible solution with the
smallest residual is kept and a warning is emitted; the resulting values
remain within the requested ranges. Invalid inputs and the absence of a screen
position satisfying geometric constraints still raise `ValueError`.

During `optimize_user_position`, the trunk flexes until the entire screen is
inside at least one reachability sphere or the permitted limit is reached.
Each sphere is centred at a shoulder joint, with a radius equal to the
maximum shoulder-to-palm distance sampled with the elbow fully extended.
Next, shoulder, elbow, forearm, and wrist flexion return the palms to their
initial positions; when bounds prevent this, the in-range posture with the
smallest residual is kept. Finally, neck coordinates are adjusted with
bounded least squares to align the line of sight with the screen centre. The
two most independent coordinates are selected and the others remain
unchanged. Metrics are calculated for the final posture. Screen-contact
reachability extends the palm-centre reach by the distance to the hand's
distal endpoint; the lumbar stage continues to use the shoulder-to-palm
radius.

The shoulder-to-palm reach radius is estimated with the elbow extended,
sampling shoulder flexion from 0° to 180° and adduction/abduction from -30° to
30°. Forearm and wrist remain within the ranges defined by the model.

By default, the line of sight is angled 15° downward from the head plane
through the eyes. The cone spans 30° vertically, from 0° to 30° below the eye
plane, independently of the line-of-sight offset. Its horizontal span is 30°
total (15° on each side). Turville et al. (1998) and Burgess-Limerick et al.
(2000) support placing displays below eye level and report postural trade-offs
at greater inclinations. The sources do not define an equivalent numerical
horizontal limit; the 30° total width is a conservative symmetric
approximation in this model, not a universal ergonomic threshold.

References: Turville et al., *Applied Ergonomics*, 29(4), 239–246 (1998),
DOI: 10.1016/S0003-6870(97)00048-3
([PubMed/NLM](https://pubmed.ncbi.nlm.nih.gov/9701537/));
Burgess-Limerick et al., *Human Factors*, 42(1), 140–150 (2000),
DOI: 10.1518/001872000779656598
([PubMed/NLM](https://pubmed.ncbi.nlm.nih.gov/10917150/)); Canadian Centre
for Occupational Health and Safety, [Monitor Positioning](https://www.ccohs.ca/oshanswers/ergonomics/office/monitor_positioning.html).

`get_view_state()` returns the incidence angle (0° is perpendicular to the
screen) and the percentages of the screen visible in the cone and reachable
by the hands. Percentages are estimated on a regular grid. It also reports
`trunk_angle_from_horizontal_deg`, `right_shoulder_flexion_deg`,
`left_shoulder_flexion_deg`, `right_elbow_flexion_deg`,
`left_elbow_flexion_deg`, `neck_base_flexion_deg`, and
`head_flexion_deg`, all in degrees. Trunk angle is measured from the
horizontal (90° when upright); `neck_base_flexion_deg` is neck flexion
relative to the trunk, and `head_flexion_deg` is head flexion relative to the
neck base. Reachability compares screen points against shoulder-centred
reach spheres extended from the palm centre to the hand's distal endpoint
using `handlength` and `palmlength`. The viewer displays only a spherical
surface oriented from the shoulder toward the screen centre, with a 45°
half-angle; this is a visual limitation and does not apply to the metric.

Additional arm and neck coordinates can be bounded through the explicit
`optimize_user_position` arguments, for example
`right_shoulder_flexion_deg=(60.0, 90.0)` and
`right_elbow_flexion_deg=(10.0, 35.0)`. These limits affect the optimized
posture used to maintain the grip. Reachability radius is instead calculated
with the arms extended, while trunk and pelvis remain in their current
posture. Screen offsets are bounded with
`screen_horizontal_offset_range_mm` and `screen_vertical_offset_range_mm`.
The constraint is checked over the entire screen surface: the screen cannot
move closer to the rider than the handlebar centre or lower than the
handlebar.

Other bike parameters and properties:

- `crank_spacing_mm` is the distance between the crank centre planes.
- `pedal_spacing_mm` is the distance between the pedals' inner faces. Its
  default is 185 mm and it is independent of `crank_spacing_mm`.
- `screen_width_mm`, `screen_height_mm`, and `screen_inclination_deg` define
  the screen dimensions and inclination.
- `screen_vertical_offset_mm` and `screen_horizontal_offset_mm` position the
  screen base relative to the handlebar centre. They must be positive and move
  the screen upward and forward.
- `screen_base` and `screen_center` return world coordinates for the screen
  base and centre in millimetres. The screen and its support are connected.
- The saddle uses rounded ellipsoidal meshes: its rear is wide and its front
  tip tapers. `seat_length_mm`, `seat_width_mm`, and `seat_height_mm` control
  its dimensions.

## STEP import

STEP import is available from `opensim_frameworks.models.equipment`. The
parser creates OpenSim bodies with mass, centre of mass, and inertia derived
from geometry and STEP material densities; surface colours are preserved in
VTP meshes. Nodes can be nested dictionaries representing assemblies. Mesh
leaves are a `Path`, or a tuple of `Path` objects when colour preservation
requires multiple files.

```python
from opensim_frameworks.models.equipment import read_step, weld_bodies

body_tree, mesh_tree = read_step(
    "cad/frame.step",
    "output/meshes",
    density_kg_m3=7850.0,
)
```

`weld_bodies(bodies, meshes=None)` combines the mass, centre of mass, and
inertia of bodies expressed in the same reference frame. If `meshes` is
provided, it must contain one path (or a list/tuple of paths) per body in the
same order. Meshes are fused by colour, opacity, and representation; scales
are applied to the resulting VTP files, which are written next to the source
meshes. The function returns only the `Body` when `meshes` is `None`;
otherwise it returns `(Body, Path | list[Path])`. Source objects are not
modified. In `HighRowPlateloaded`, components that must move independently
remain separate bodies connected by `WeldJoint`.

```python
combined_body, fused_meshes = weld_bodies(
    [body_a, body_b],
    ["meshes/a.vtp", ("meshes/b1.vtp", "meshes/b2.vtp")],
)
```

STEP-tree utilities weld a complete tree or extract one component with its
descendants:

```python
from opensim_frameworks.models.equipment import (
    extract_step_body_and_mesh,
    read_step,
    weld_step_body_mesh_tree,
)

body_tree, mesh_tree = read_step(
    "cad/frame.step",
    "output/meshes",
    density_kg_m3=7850.0,
)
part_body, part_meshes = extract_step_body_and_mesh(
    body_tree,
    mesh_tree,
    "part_name",
    remove=False,
)
frame_body, frame_meshes = weld_step_body_mesh_tree(body_tree, mesh_tree)
```

`extract_step_body_and_mesh()` removes the extracted node from both trees by
default and prunes empty ancestor assemblies; set `remove=False` to preserve
the input trees. `read_vtp_vertices()` reads vertex coordinates from one or
more VTP files. `body_mass_center_array()` converts an OpenSim body's centre
of mass to a NumPy array, while `get_referenced_meshes()` returns the meshes
attached to a model and their referenced file names. These utilities are
also available from the top-level `opensim_frameworks` namespace.

## Composition with `Framework`

`Framework` is a dictionary-like container of separate OpenSim models
associated with names. Members can be accessed with either
`framework["user"]` or, when the name is not reserved, `framework.user`.

```python
from opensim_frameworks import Bike

framework = Bike().fit_user()
state = framework.initSystem()
composite = framework.composite
standalone_model = framework.merge_to_model()
```

`initSystem()` rebuilds and initializes a composite snapshot; models stored
in the framework remain separate. Direct changes to a member, for example
`framework["user"].set_neck_flexion(30)`, update that model. A subsequent
`merge_to_model()` reads the current member states and creates a new,
independent `opensim.Model`. An already initialized composite snapshot is not
updated automatically; call `initSystem()` again to rebuild it. `forward()`
and `settle_under_gravity()` synchronize results back to member models. After
manually changing the composite state, call `update_state()` to transfer
coordinates and speeds. The `only`, `exclude`, and `include_couplings`
parameters of `merge_to_model()` select which members and couplings to
include.

Framework couplings are recreated in composite snapshots and do not directly
modify member models. The main operations are:

- `couple_weld()` to weld two frames;
- `couple_point()` to make two frame points coincide;
- `couple_coordinate()` to drive a coordinate from other coordinates;
- `couple_joint()` to connect a joint in an otherwise ground-rooted model to
  a frame;
- `add_force()`, `add_constraint()`, and `add_controller()` to add OpenSim
  components to the composite;
- `uncouple()` to remove a coupling.

Component references are `(model_name, OpenSim_path)` tuples, for example
`("user", "/bodyset/hand_r")`. All models in a framework must use consistent
gravity. `set_gravity((0.0, -9.81, 0.0))` sets a common gravity vector in m/s².

`forward(final_time)` integrates to the specified absolute time in seconds.
`settle_under_gravity(duration_s)` integrates for the specified duration from
the current time but does not check whether the system has reached
equilibrium. Integration through `opensim.Manager` explicitly rejects models
containing an `opensim.FreeJoint`.

## Visualization and export

```python
from opensim_frameworks import VTKViewer

viewer = VTKViewer()
window = viewer.show(framework, motion="motion.sto", loop=True, fps=30.0)
viewer.save_snapshot(
    framework,
    path="output/pose.svg",
    motion="motion.sto",
    time_s=1.5,
    camera_position=(2.0, 1.5, 0.0),
    camera_focal_point=(0.0, 1.0, 0.0),
    camera_view_up=(0.0, 1.0, 0.0),
    show_ground=False,
    show_axes=True,
    show_muscles=False,
    show_markers=True,
    show_reachability=False,
    show_vision_cone=True,
)
viewer.save_animation(
    framework,
    "output/motion.mp4",
    motion="motion.sto",
    camera_position=(2.0, 1.5, 0.0),
    camera_focal_point=(0.0, 1.0, 0.0),
    camera_view_up=(0.0, 1.0, 0.0),
    show_ground=False,
    show_axes=True,
    show_muscles=False,
    show_markers=True,
    show_reachability=False,
    show_vision_cone=True,
)
```

`show()` opens the interactive viewer and returns a playback-control window.
It supports OpenSim models and `Framework` objects; a framework is rendered
as a composite snapshot. Motion tables can be supplied as `.mot`/`.sto` paths
or as an `opensim.TimeSeriesTable` with at least two samples. The window
provides playback, stop, seek, and repeat controls; preset camera views; and
visibility toggles for ground, muscles, markers, XYZ axes, the visual cone,
and reachability regions. Unavailable overlay controls are disabled when a
model does not contain those overlays.

`save_snapshot()` exports an **SVG** or **PNG** image without opening a
window, based on the destination extension. For PNG output, `dpi` sets the
DPI recorded in the metadata (300 by default) and does not change pixel
dimensions; use `size` to configure those dimensions. `frame` selects a row
from a motion table and `time_s` interpolates the pose at the requested time.
Without a motion table, the current pose is used; if both are supplied,
`frame` takes precedence over `time_s`. Export does not modify the source
model pose.

`camera_position`, `camera_focal_point`, and `camera_view_up` configure the
camera and focal point in world coordinates (metres). If omitted, the camera
is fitted automatically. The Boolean options `show_ground`, `show_axes`,
`show_muscles`, `show_markers`, `show_reachability`, and `show_vision_cone`
control the ground grid, XYZ axes, muscle paths, markers, and overlays; they
are all enabled by default. Other image extensions are not supported.

`save_animation()` exports an off-screen **MP4** and restores the model's
coordinate values after rendering. It accepts the same camera and visibility
options as `save_snapshot()`. `size` configures frame dimensions, which must
be positive even integers. The frame rate is derived from the motion time
interval, and the motion is sampled uniformly. Columns matching model
coordinates are applied, including `/coordinate/value` labels in a states
table. The top-level `save_animation(model, table, path, ...)` function also
accepts an OpenSim model, a table, or a motion path. `VTKViewer.save_animation()`
additionally accepts a `Framework`. If any camera option is supplied, omitted
vectors retain their current automatic values. `User.show()` and
`Framework.show()` also provide a shortcut to open the viewer.

## Tests

The repository configures pytest through `pyproject.toml`; tests are grouped
by package area (`framework`, `models`, `viewer`, and `package`). After
installing the test extra, run:

```powershell
python -m pytest
```

In VS Code, select the interpreter containing OpenSim and configure pytest as
the Python test framework.

## Public API

The top-level `opensim_frameworks` namespace exposes `User`, `Bike`,
`IRON_DENSITY_KG_M3`, `Framework`, `save_animation`, `VTKViewer`,
`VTKVisualizer`, `MotionData`, `MotionPlayer`, `PlayerWindow`, `DEFAULT_SIZE`,
and `VIEW_NAMES`.

`Equipment`, `Screen`, `read_step`, `weld_bodies`,
`extract_step_body_and_mesh`, and `weld_step_body_mesh_tree` are available
from `opensim_frameworks.models.equipment`. `read_vtp_vertices`,
`body_mass_center_array`, and `get_referenced_meshes` are also exposed from
the top-level namespace. `VisualErgonomicsFramework` is available from
`opensim_frameworks.framework`. Geometric primitives and OpenSim body
builders are exposed from `opensim_frameworks.bodies`, including `Box`,
`Cylinder`, `Disk`, `Ellipsoid`, `Sphere`, `Tube`, and `mass_properties`.
`weld_bodies` combines rigidly connected OpenSim bodies. Primitives use SI
units, and their builders provide helpers for calculating mass properties
and adding welded or articulated bodies to OpenSim models.
