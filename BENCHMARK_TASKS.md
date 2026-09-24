# In-Context Robot Learning Benchmark

## 1. Benchmark Goal

This benchmark evaluates whether a robot policy can infer **task intent from a demonstration** and execute the same intent under different rollout conditions.

The demonstration and rollout always correspond to the **same task intent**, but the concrete execution may differ because of:

- object layout changes,
- target layout changes,
- distractor objects,
- object appearance changes,
- environment changes,
- articulation / mechanism changes,
- different geometric constraints.

The central question is:

> Can the policy separate **what should be achieved** from **how the demonstrator happened to achieve it**?

Formally,

\[
D \rightarrow z_{\text{intent}}
\]

and

\[
(o_t, z_{\text{intent}}) \rightarrow a_t
\]

where the demonstration \(D\) specifies the task intent and the rollout observation \(o_t\) determines the concrete execution.

---

## 2. Evaluation Axes

### 2.1 Task Split

**Seen Task**

The concrete task appears during training.

Example:

> Banana → Blue Bowl

**Unseen Task**

The task belongs to a known task family, but the concrete object-target combination, object category, target category, mechanism, or ordering is held out from training.

Example:

> Pear → Purple Bowl

---

### 2.2 Execution Difficulty

Task novelty and execution difficulty should be treated as two independent axes.

| Level | Setting | Description |
|---|---|---|
| E0 | Matched Layout | Prompt and rollout have similar layouts |
| E1 | Layout Shift | Relevant objects appear at different positions |
| E2 | Distractors | Additional irrelevant objects are introduced |
| E3 | Environment Shift | Background / table / room changes |
| E4 | Geometry / Mechanism Shift | The same intent requires a substantially different motion |

The most important setting is **E4**, because successful execution provides strong evidence that the policy is not simply copying the demonstrated trajectory.

---

# 3. Task Families

---

# F1. Object-to-Receptacle Placement

## Intent

Place a specified object \(A\) into or onto a specified target \(B\).

\[
\boxed{\text{Put object } A \text{ into/on target } B}
\]

This family mainly evaluates:

- object binding,
- target binding,
- robustness to layout changes,
- robustness to distractors,
- novel object-target combinations.

## Seen Tasks

| ID | Task |
|---|---|
| F1-S01 | Put the banana into the blue bowl |
| F1-S02 | Put the banana into the red bowl |
| F1-S03 | Put the apple into the blue bowl |
| F1-S04 | Put the orange into the green bowl |
| F1-S05 | Put the red block onto the yellow tray |
| F1-S06 | Put the tomato into the white basket |
| F1-S07 | Put the sponge into the black box |
| F1-S08 | Put the toy car into the red box |
| F1-S09 | Put the wooden block onto the metal plate |
| F1-S10 | Put the spoon into the white cup |

## Unseen Tasks

| ID | Task |
|---|---|
| F1-U01 | Put the pear into the purple bowl |
| F1-U02 | Put the apple into the purple bowl |
| F1-U03 | Put the peach into the green bowl |
| F1-U04 | Put the lemon into the red bowl |
| F1-U05 | Put the blue block into the white basket |
| F1-U06 | Put the cucumber into the blue bowl |
| F1-U07 | Put the soap onto the yellow tray |
| F1-U08 | Put the toy duck into the black box |
| F1-U09 | Put the golf ball onto the metal plate |
| F1-U10 | Put the fork into the purple cup |

## Hard Variants

A rollout may additionally contain:

- displaced object and target,
- multiple distractor objects,
- multiple visually similar objects,
- multiple instances of the target category,
- an unseen tabletop / room.

**Note:**  
"Move one object" and "move all objects of a category" should be treated as different task intents. If `move-all` is evaluated, corresponding examples should also appear during training.

---

# F2. Articulated Object State Change

## Intent

Change an articulated object from one state to another.

\[
\boxed{
q_{\text{object}}:
\text{closed} \leftrightarrow \text{open}
}
\]

The key challenge is that the same semantic goal may require completely different movements under different articulation geometries.

## Seen Tasks

| ID | Task |
|---|---|
| F2-S01 | Open a red cabinet door with a left-side hinge |
| F2-S02 | Close a blue cabinet door with a left-side hinge |
| F2-S03 | Open an upward-opening cabinet door |
| F2-S04 | Close an upward-opening cabinet door |
| F2-S05 | Pull open a shallow horizontal drawer |
| F2-S06 | Push closed a shallow horizontal drawer |
| F2-S07 | Open a left-sliding cabinet panel |
| F2-S08 | Close a left-sliding cabinet panel |
| F2-S09 | Open a top-hinged storage-box lid |
| F2-S10 | Close a top-hinged storage-box lid |

## Unseen Tasks

| ID | Task |
|---|---|
| F2-U01 | Open a green cabinet door with a right-side hinge |
| F2-U02 | Close a white cabinet door with a right-side hinge |
| F2-U03 | Open a bottom-hinged oven door downward |
| F2-U04 | Close a bottom-hinged oven door upward |
| F2-U05 | Pull open a deep horizontal drawer |
| F2-U06 | Push closed a deep horizontal drawer |
| F2-U07 | Open a vertically sliding panel downward |
| F2-U08 | Close a vertically sliding panel upward |
| F2-U09 | Open a side-hinged toolbox |
| F2-U10 | Close a laptop screen |

## Key Evaluation

A particularly important evaluation setting is:

**Prompt**

> Grasp the handle and move left to open a left-hinged door.

**Rollout**

> The hinge is below the door, so the correct motion is downward and backward.

Successful execution demonstrates that the policy extracts:

\[
\text{intent} = \text{open}
\]

rather than:

\[
\text{trajectory} = \text{move left}
\]

---

# F3. Container-to-Container Transfer

## Intent

Transfer the contents of one container into another container.

\[
\boxed{
\text{Transfer contents of } A \text{ into } B
}
\]

This family evaluates functional motion, target alignment, and orientation-conditioned execution.

## Seen Tasks

| ID | Task |
|---|---|
| F3-S01 | Pour from a white paper cup into a blue bowl |
| F3-S02 | Pour from a red paper cup into a green bowl |
| F3-S03 | Pour from a blue paper cup into a white cup |
| F3-S04 | Pour from a small plastic cup into a red bowl |
| F3-S05 | Pour from a transparent cup into a yellow cup |
| F3-S06 | Pour from a small water bottle into a blue cup |
| F3-S07 | Pour from a red measuring cup into a white bowl |
| F3-S08 | Pour from a plastic bottle into a green bowl |
| F3-S09 | Pour from a white mug into a black bowl |
| F3-S10 | Pour from a blue pitcher into a red cup |

## Unseen Tasks

| ID | Task |
|---|---|
| F3-U01 | Pour from a Coke can into a green bowl |
| F3-U02 | Pour from a Sprite can into a purple bowl |
| F3-U03 | Pour from a yellow mug into a red bowl |
| F3-U04 | Pour from a metal cup into a blue bowl |
| F3-U05 | Pour from a milk carton into a white cup |
| F3-U06 | Pour from a juice bottle into a purple cup |
| F3-U07 | Pour from a handled pitcher into a green cup |
| F3-U08 | Pour from a rectangular carton into a red cup |
| F3-U09 | Pour from a small glass bottle into a yellow bowl |
| F3-U10 | Pour from a wide-mouth container into a white bowl |

## Success Definition

If empty containers are used to simplify data collection, success should be defined geometrically, for example:

\[
\text{source above target}
\]

\[
\text{tilt angle} > \theta
\]

for a minimum duration \(\Delta t\).

Harder variants may vary:

- target location,
- required tilt direction,
- container orientation,
- container handle position,
- target opening size.

---

# F4. Constrained Insertion

## Intent

Insert an object into a geometrically constrained receptacle.

\[
\boxed{
\text{Insert } A \text{ into receptacle } B
}
\]

Unlike ordinary pick-and-place, the final orientation and approach direction are essential parts of the task.

## Seen Tasks

| ID | Task |
|---|---|
| F4-S01 | Insert a red book into a vertical bookshelf slot |
| F4-S02 | Insert a blue book into a vertical bookshelf slot |
| F4-S03 | Insert a blue plate into a plate rack |
| F4-S04 | Insert a red cylindrical peg into a circular hole |
| F4-S05 | Insert a square peg into a square hole |
| F4-S06 | Insert a spoon into a utensil holder |
| F4-S07 | Insert a card into a wide card slot |
| F4-S08 | Insert a wooden strip into a horizontal slot |
| F4-S09 | Insert a thin rod into a circular tube |
| F4-S10 | Insert a rectangular block into a rectangular socket |

## Unseen Tasks

| ID | Task |
|---|---|
| F4-U01 | Insert a bread slice into a toaster slot |
| F4-U02 | Insert a yellow book into an angled bookshelf slot |
| F4-U03 | Insert a white plate into a horizontally oriented rack |
| F4-U04 | Insert a blue cylindrical peg into an angled circular hole |
| F4-U05 | Insert a triangular peg into a triangular hole |
| F4-U06 | Insert a knife into a knife-block slot |
| F4-U07 | Insert an access card into a narrow card slot |
| F4-U08 | Insert a puzzle piece into its matching socket |
| F4-U09 | Insert a toothbrush into a toothbrush-holder opening |
| F4-U10 | Insert a battery into a battery compartment |

## Hard Variants

The rollout receptacle can be:

- rotated,
- tilted,
- translated,
- narrower than the prompt receptacle,
- visually different,
- surrounded by distractors.

The intended behavior should remain invariant while the approach trajectory changes.

---

# F5. Rule-Conditioned Sorting

## Intent

Infer a grouping rule from the demonstration and apply the same rule to new objects.

\[
\boxed{
x \in C_i \Rightarrow B_i
}
\]

This family is particularly useful for evaluating genuine in-context learning because the same rollout observation can correspond to different tasks depending entirely on the demonstration.

## Seen Tasks

| ID | Task |
|---|---|
| F5-S01 | Red balls → left box; blue balls → right box |
| F5-S02 | Round objects → left box; square objects → right box |
| F5-S03 | Fruits → green basket; toys → red basket |
| F5-S04 | Large objects → left box; small objects → right box |
| F5-S05 | Red blocks → front box; green blocks → rear box |
| F5-S06 | Cups → left box; bowls → right box |
| F5-S07 | Metal objects → left box; plastic objects → right box |
| F5-S08 | Balls → left box; blocks → right box |
| F5-S09 | Objects with handles → left box; objects without handles → right box |
| F5-S10 | Warm-colored objects → left box; cool-colored objects → right box |

## Unseen Tasks

| ID | Task |
|---|---|
| F5-U01 | Batteries → upper box; erasers → lower box |
| F5-U02 | Triangular objects → left box; cylindrical objects → right box |
| F5-U03 | Stationery → blue box; food → yellow box |
| F5-U04 | Long objects → upper box; short objects → lower box |
| F5-U05 | Yellow objects → left box; purple objects → right box |
| F5-U06 | Plates → upper box; bottles → lower box |
| F5-U07 | Wooden objects → front box; soft objects → rear box |
| F5-U08 | Tools → red box; utensils → blue box |
| F5-U09 | Rollable objects → upper box; non-rollable objects → lower box |
| F5-U10 | Food → basket; containers → tray |

## Important Protocol

A sorting demonstration should contain multiple examples so that the underlying rule is identifiable.

For example, the same rollout scene may be paired with:

**Prompt A**

> Sort by color.

**Prompt B**

> Sort by shape.

**Prompt C**

> Sort by semantic category.

The rollout observation can remain identical while the correct action changes entirely according to the context.

---

# F6. Ordered Stacking and Composition

## Intent

Construct a stack, optionally following an ordering rule demonstrated in the prompt.

Simple version:

\[
\boxed{\text{Form a stack}}
\]

Ordered version:

\[
\boxed{
\text{Form a stack in the demonstrated order}
}
\]

This family evaluates sequence understanding, object ordering, and compositional task intent.

## Seen Tasks

| ID | Task |
|---|---|
| F6-S01 | Stack three gray bowls |
| F6-S02 | Stack three red paper cups |
| F6-S03 | Stack three wooden blocks into a tower |
| F6-S04 | Stack three white plates |
| F6-S05 | Stack red → green → blue blocks from bottom to top |
| F6-S06 | Stack blue → white → red cups from bottom to top |
| F6-S07 | Stack large → medium → small wooden blocks |
| F6-S08 | Nest small → medium → large bowls |
| F6-S09 | Stack four colored blocks according to the demonstrated order |
| F6-S10 | Stack red → blue → red blocks in the demonstrated order |

## Unseen Tasks

| ID | Task |
|---|---|
| F6-U01 | Stack three blue books |
| F6-U02 | Stack three green cups |
| F6-U03 | Stack three cans into a tower |
| F6-U04 | Stack three boxes with different sizes |
| F6-U05 | Stack yellow → purple → white blocks from bottom to top |
| F6-U06 | Stack green → red → blue bowls from bottom to top |
| F6-U07 | Stack large → medium → small books |
| F6-U08 | Nest small → medium → large paper cups |
| F6-U09 | Stack four differently colored books according to the demonstrated order |
| F6-U10 | Stack three differently colored bowls according to the prompt order |

## Hard Variants

Prompt:

```text
Initial layout:
Red   Blue   Green

Desired stack:
Red → Green → Blue