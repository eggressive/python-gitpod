"""
    Calls the Together API to generate a response based on the given prompt.
    
    Args:
        prompt (str): The input prompt for the model.
        add_inst (bool, optional): Whether to add an instruction tag to the prompt. Defaults to True.
        model (str, optional): The name of the model to use. Defaults to "togethercomputer/llama-2-7b-chat".
        temperature (float, optional): The temperature parameter for controlling the randomness of the output. Defaults to 0.0.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.
        verbose (bool, optional): Whether to print additional information. Defaults to False.
        url (str, optional): The URL of the Together API. Defaults to the value from the environment variable 'DLAI_TOGETHER_API_BASE'.
        headers (dict, optional): The headers for the API request. Defaults to the authorization header with the value from the environment variable 'TOGETHER_API_KEY'.
        base (int, optional): The base number of seconds to wait between API call attempts. Defaults to 2.
        max_tries (int, optional): The maximum number of attempts to call the API. Defaults to 3.
    
    Returns:
        str: The generated response from the model.
"""

import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv, find_dotenv
import warnings
import requests
import json
import time

# Initialize global variables
_ = load_dotenv(find_dotenv())
# warnings.filterwarnings('ignore')
url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/inference"
headers = {
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
    "Content-Type": "application/json",
}


import time


def llama(
    prompt,
    add_inst=True,
    model="togethercomputer/llama-2-7b-chat",
    temperature=0.0,
    max_tokens=1024,
    verbose=False,
    url=url,
    headers=headers,
    base=2,  # number of seconds to wait
    max_tries=3,
):

    if add_inst:
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Allow multiple attempts to call the API in case of downtime.
    # Return provided response to user after 3 failed attempts.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()["output"]["choices"][0]["text"]
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(
                f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again."
            )
            time.sleep(wait_seconds[num_tries])

    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response
