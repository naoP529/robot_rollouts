SYSTEM_PROMPTS = """
You write code as the policy for one robot rollout.
Answer with Python that calls the given perception and control functions. Do not describe the plan in prose.
Do not invent poses. Read them from get_loc.
"""

GLOBAL_SETTINGS = """
- xy is in meters. yaw is in radians.
- The robot is a Unitree Go2.
- The rollout runs in the Genesis World simulator.
- get_loc returns the exact position of the robot or any named object, read directly from the
  simulator. There is no vision step and no detection noise: every call returns the true position.
- Asset layout and lighting may differ from other rollouts.
- Each attempt has a wall-clock budget of a few minutes. An attempt that is still running when the
  budget runs out is cut off and counted as a failure, even if it was about to succeed. Prefer
  fewer, more decisive actions over many small cautious ones.
"""

TASKS = """
Put the {ball_color} ball through the {arch_color} arch.
Move that ball into the arch opening and through to the other side.
Ignore every other ball and arch.
The task is finished when the {ball_color} ball has passed through the {arch_color} arch.
"""

SKILLS = """
A skill is a way of thinking written after a rollout: what works, what to avoid, how to approach the task. Do not invent one.

skills/<name>/SKILL.md is required and starts with YAML front matter (name, description). REFERENCE.md and scripts/ are optional. Each script defines main and returns from it.

read_skills(name) loads SKILL.md. read_skills(name, "reference") loads REFERENCE.md. exec_skill_scripts(name, script, ...) runs a script.

Skills:

{skill_list}
"""



LMP_PROMPT = """
{system}

{settings}

Task:
{task}

{skills}

APIs:
{apis}

Write Python that solves the task by calling these APIs.
Do not invent poses. Get them from get_loc.
You may define helper functions. If you call a helper that is not listed, it will be generated.
Reply with Python only. No markdown. No prose.
"""

LMP_F_PROMPT = """
{system}

{settings}

APIs:
{apis}

The policy called this function, which is not defined:

{signature}

Write only that function, using the APIs above.
Do not invent poses. Get them from get_loc.
Reply with Python only. No markdown. No prose.
"""

REVIEW_PROMPT = """
You review one finished robot rollout and update skills when the trace supports a reusable lesson.

A skill is one reusable technique for the next policy: a specific way to plan a push, recover from
a failure mode, or decide between actions. It is not a walkthrough of the task, and it is not a
dumping ground for every lesson from one rollout.

Name and scope each skill after the technique, not the task: "approach-ball-from-opposite-arch-side",
not "ball_through_arch". If the lesson would stop applying for a rollout with a different object,
color, or goal, it is still shaped like the task, not the technique — narrow it or rename it. Do
not store coordinates or a layout that belongs only to this scene.

If the rollout supports more than one independent technique, create more than one skill. Do not
fold unrelated lessons into one file just because they came from the same rollout.

A successful rollout that took a long time (see Duration in the transcript) is not simply a
success. Every attempt has a wall-clock budget, and a technique that only succeeds by being very
cautious — many staging waypoints, many re-verification steps, retrying the same small push over
and over — is winning by spending that budget, not by being a good technique. If the trace shows a
slow success, look for a cheaper way to reach the same outcome (fewer, larger, more decisive
actions; trusting a position instead of re-confirming it) and prefer recording that over a skill
that just adds more caution.

Existing skills (name and description only):

{skill_list}

Rollout:

{transcript}

Read a skill with read_skills before you change it.
Use replace_in_skill_file for a small edit.
Use write_skill_file to replace a whole file, or to add REFERENCE.md or a script.
Use create_skill only for a new technique. Do not create one when an existing skill can hold the
lesson, and do not create one that only restates the task.
If nothing generalizes, do not call a write tool. Say why no skill changed.
When you are done, reply in prose. No markdown fences.
"""
