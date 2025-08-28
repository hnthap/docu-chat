import asyncio
import getpass
import logging
from pathlib import Path


def print_dict(d):
    """
    Prints a dictionary with keys aligned for readability.

    Args:
        d (dict): The dictionary to print.
    """
    max_key_len = max(map(len, list(d.keys())))
    for key, value in d.items():
        print(f'{key}{" " * (max_key_len - len(key))}: {value}')

    
def beauty_input(prompt):
    """
    Reads non-empty input from the user, retrying on empty input.

    Args:
        prompt (str): The input prompt string.

    Returns:
        str: The user's non-empty input.
    """
    while True:
        try:
            s = input(prompt).strip()
            if s:
                return s
        except EOFError:
            pass
        

async def abeauty_input(prompt):
    return await asyncio.to_thread(beauty_input, prompt)


async def agetpass(prompt, stream=None):
    return await asyncio.to_thread(getpass.getpass, prompt, stream)
    

def setup_logging():
    """Configures logging to file and console."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'rag_app.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            # logging.StreamHandler() # To also log to console
        ]
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
