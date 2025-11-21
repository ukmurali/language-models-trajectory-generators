# main.py (UPDATED with dynamic LLM object selection)

import os
import sys
import argparse
import functools
import logging
import multiprocessing
import traceback
from multiprocessing import Process, Pipe
from io import StringIO
from contextlib import redirect_stdout

import openai
import torch

import models
import config
from lang_sam import LangSAM
from api import API
from env import run_simulation_environment
from prompts.main_prompt import MAIN_PROMPT
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, FAIL, ENDC

print = functools.partial(print, flush=True)

sys.path.append("./XMem/")
from XMem.model.network import XMem

# ---------------------------
# System-level instructions
# ---------------------------
SYSTEM_INSTRUCTIONS = """
IMPORTANT INSTRUCTIONS FOR THE MODEL (READ CAREFULLY):

1) The function `detect_object(name)` returns a dictionary:
   - 'label'
   - 'position'
   - 'grasp_hover'
   - 'grasp_touch'
   - 'orientation'
   - 'masks'

2) Use ONLY the helper functions:
   detect_object(name)
   execute_trajectory(traj)
   open_gripper()
   close_gripper()
   task_completed()

3) Trajectory format:
   - (pos, orient)
   - {"pos":[x,y,z], "orient":[qx,qy,qz,qw]}
   - [x,y,z]

4) Workflow:
   - ALWAYS call detect_object first.
   - Build trajectory using returned values.
   - Then execute trajectory.
   - Then close_gripper.
   - Then lift.
   - Then call task_completed().

5) Safety:
   - Keep positions inside robot workspace.
"""

def make_system_prompt(task_text):
    base = SYSTEM_INSTRUCTIONS + "\n\n" + MAIN_PROMPT
    base = base.replace("[INSERT EE POSITION]", str(config.ee_start_position))
    base = base.replace("[INSERT TASK]", task_text)
    return base


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            print("[WARN] OPENAI_API_KEY not set!")
        client = openai.OpenAI(api_key=openai.api_key)

        parser = argparse.ArgumentParser(description="Main Program.")
        parser.add_argument("-lm", "--language_model",
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                            default="gpt-4o")
        parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer")
        parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default")
        args = parser.parse_args()

        logger = multiprocessing.log_to_stderr()
        logger.setLevel(logging.INFO)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: " + str(device))

        torch.set_grad_enabled(False)

        logger.info("Loading LangSAM and XMem…")
        langsam_model = LangSAM()
        xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
        logger.info("✔ Models loaded.")

        main_connection, env_connection = Pipe()
        api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)

        # Safe exposed functions for LLM code execution
        def detect_object_fn(name): return api.detect_object(name)
        def detect_all_objects_fn(labels): return api.detect_all_objects(labels)
        def execute_trajectory_fn(traj): return api.execute_trajectory(traj)
        def open_gripper_fn(): return api.open_gripper()
        def close_gripper_fn(): return api.close_gripper()
        def task_completed_fn(): return api.task_completed()

        exec_globals = {
            "detect_object": detect_object_fn,
            "detect_all_objects": detect_all_objects_fn,
            "execute_trajectory": execute_trajectory_fn,
            "open_gripper": open_gripper_fn,
            "close_gripper": close_gripper_fn,
            "task_completed": task_completed_fn,
            "print": print,
            "__builtins__": __builtins__,
        }

        # Start simulator
        env_process = Process(target=run_simulation_environment,
                              args=(args, env_connection, logger))
        env_process.start()

        [env_msg] = main_connection.recv()
        logger.info(env_msg)

        # ---------------------------
        # MAIN LOOP
        # ---------------------------
        command = input("Enter a command: ")
        api.command = command
        logger.info(PROGRESS + "STARTING TASK…" + ENDC)

        # ---------------------------
        # 🔥 LLM OBJECT SELECTION HERE
        # ---------------------------
        scene_description = "table has one red cup, one yellow banana, one black phone"
        segmentation_name = api.choose_object_with_llm(scene_description, command)
        print("LLM selected object:", segmentation_name)

        exec_globals["segmentation_name"] = segmentation_name

        system_prompt = make_system_prompt(command) + f"\nSelected object: {segmentation_name}\n"

        messages = []
        error = False

        logger.info(PROGRESS + "Generating ChatGPT output…" + ENDC)
        messages = models.get_chatgpt_output(client, args.language_model, system_prompt, messages, "system")
        logger.info(OK + "Finished ChatGPT output." + ENDC)

        # ---------------------------
        # EXEC LOOP
        # ---------------------------
        while True:
            while not api.completed_task:
                new_prompt = ""
                content = messages[-1]["content"]

                # extract python blocks
                if "```python" in content:
                    blocks = content.split("```python")
                    block_number = 0

                    for block in blocks:
                        if "```" in block:
                            block_number += 1
                            code = block.split("```")[0]

                            try:
                                f = StringIO()
                                with redirect_stdout(f):
                                    exec(code, exec_globals)
                                output = f.getvalue()

                                if output.strip():
                                    new_prompt += PRINT_OUTPUT_PROMPT.replace(
                                        "[INSERT PRINT STATEMENT OUTPUT]", output
                                    )
                                    error = True
                            except Exception:
                                err = traceback.format_exc()
                                new_prompt += ERROR_CORRECTION_PROMPT \
                                    .replace("[INSERT BLOCK NUMBER]", str(block_number)) \
                                    .replace("[INSERT ERROR MESSAGE]", err)
                                error = True

                if error:
                    api.completed_task = False
                    api.failed_task = False

                if not api.completed_task:
                    if api.failed_task:
                        logger.info(FAIL + "FAILED TASK!" + ENDC)
                        new_prompt += TASK_SUMMARY_PROMPT

                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")

                        new_prompt = make_system_prompt(command) + "\n"
                        new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])

                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, [], "system")

                        api.failed_task = False

                    else:
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        error = False

            logger.info(OK + "FINISHED TASK!" + ENDC)

            # NEW COMMAND
            new_command = input("Enter a new command: ")
            api.command = new_command

            # Run object-selection again for new command
            new_scene = "table has one red cup, one yellow banana, one black phone"
            segmentation_name = api.choose_object_with_llm(new_scene, new_command)
            exec_globals["segmentation_name"] = segmentation_name

            system_prompt = make_system_prompt(new_command) + f"\nSelected object: {segmentation_name}\n"
            messages = models.get_chatgpt_output(client, args.language_model, system_prompt, [], "system")
            api.completed_task = False

    except KeyboardInterrupt:
        print("Exiting…")

    except Exception as e:
        print("Exception:", e)
        traceback.print_exc()

    finally:
        try:
            main_connection.close()
        except:
            pass
        if 'env_process' in locals() and env_process.is_alive():
            env_process.terminate()
            env_process.join()
        print("Program exited cleanly.")
