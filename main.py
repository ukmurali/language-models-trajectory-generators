import openai
import torch
import os
import sys
import argparse
import traceback
import multiprocessing
import logging
import functools
import models
import config
from lang_sam import LangSAM
from multiprocessing import Process, Pipe
from io import StringIO
from contextlib import redirect_stdout
from api import API
from env import run_simulation_environment
from prompts.main_prompt import MAIN_PROMPT
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, FAIL, ENDC

# Add XMem path
sys.path.append("./XMem/")
print = functools.partial(print, flush=True)

from XMem.model.network import XMem

if __name__ == "__main__":
    try:
        # 🔐 Set up OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(
            api_key="" \
            )

        # ⚙️ Parse CLI arguments
        parser = argparse.ArgumentParser(description="Main Program.")
        parser.add_argument(
            "-lm", "--language_model",
            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            default="gpt-4o", help="select language model"
        )
        parser.add_argument(
            "-r", "--robot",
            choices=["sawyer", "franka"],
            default="sawyer",
            help="select robot"
        )
        parser.add_argument(
            "-m", "--mode",
            choices=["default", "debug"],
            default="default",
            help="select mode to run"
        )
        args = parser.parse_args()

        print('args value', args)

        # 🧠 Logging setup
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(logging.INFO)

        # ⚡ Device selection
        if torch.cuda.is_available():
            logger.info("Using GPU.")
            device = torch.device("cuda")
        else:
            logger.info("CUDA not available. Using CPU.")
            device = torch.device("cpu")

        torch.set_grad_enabled(False)

        # 🤖 Load models
        logger.info("Loading LangSAM and XMem models...")
        langsam_model = LangSAM()
        xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
        logger.info("✅ Models loaded.")

        # 🔌 Create pipe & API
        main_connection, env_connection = Pipe()
        api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)
        # Expose API methods to exec() global scope
        detect_object = api.detect_object
        execute_trajectory = api.execute_trajectory
        open_gripper = api.open_gripper
        close_gripper = api.close_gripper
        task_completed = api.task_completed
        task_failed = api.task_failed

        # 🚀 Start environment process
        env_process = Process(
            target=run_simulation_environment,
            name="EnvProcess",
            args=(args, env_connection, logger)
        )
        env_process.start()

        # 🧠 Receive initial environment message
        [env_connection_message] = main_connection.recv()
        logger.info(env_connection_message)

        # 🧠 Task command input
        command = input("Enter a command: ")
        api.command = command

        logger.info(PROGRESS + "STARTING TASK..." + ENDC)

        messages = []
        error = False

        new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)) \
            .replace("[INSERT TASK]", command)

        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "system")
        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

        # 🧠 Main loop
        while True:
            while not api.completed_task:
                new_prompt = ""

                # 🔍 Extract code blocks from ChatGPT output
                if len(messages[-1]["content"].split("```python")) > 1:
                    code_block = messages[-1]["content"].split("```python")
                    block_number = 0

                    for block in code_block:
                        if not error and len(block.split("```")) > 1:
                            code = block.split("```")[0]
                            block_number += 1
                            try:
                                f = StringIO()
                                with redirect_stdout(f):
                                    exec(code)
                            except Exception:
                                error_message = traceback.format_exc()
                                new_prompt += ERROR_CORRECTION_PROMPT \
                                                  .replace("[INSERT BLOCK NUMBER]", str(block_number)) \
                                                  .replace("[INSERT ERROR MESSAGE]", error_message) + "\n"
                                error = True
                            else:
                                s = f.getvalue()
                                error = False
                                if s != "" and len(s) < 2000:
                                    new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]",
                                                                              s) + "\n"
                                    error = True

                if error:
                    api.completed_task = False
                    api.failed_task = False

                if not api.completed_task:
                    if api.failed_task:
                        logger.info(FAIL + "FAILED TASK! Generating summary..." + ENDC)

                        new_prompt += TASK_SUMMARY_PROMPT + "\n"

                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                        logger.info(PROGRESS + "RETRYING TASK..." + ENDC)

                        new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)) \
                                         .replace("[INSERT TASK]", command) + "\n"
                        new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])

                        messages = []
                        error = False

                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages,
                                                             "system")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                        api.failed_task = False
                    else:
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
                        error = False

            logger.info(OK + "FINISHED TASK!" + ENDC)
            new_prompt = input("Enter a new command: ")
            logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
            logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
            api.completed_task = False

    except KeyboardInterrupt:
        print("🛑 KeyboardInterrupt received. Exiting...")

    except Exception as e:
        print("❌ Exception in main loop:", e)
        traceback.print_exc()

    finally:
        # 🧹 Graceful cleanup
        if 'main_connection' in locals():
            try:
                main_connection.close()
                print("🔌 main_connection closed.")
            except Exception:
                pass

        if 'env_process' in locals() and env_process.is_alive():
            print("🧹 Terminating EnvProcess...")
            env_process.terminate()
            env_process.join(timeout=5)
            print("✅ EnvProcess terminated.")

        print("🏁 Program exiting cleanly.")
