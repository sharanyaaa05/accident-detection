import ast
import time

def get_timestamp_in_millis() -> int:
    return int(time.time() * 1000)

def convert_string_to_array(s):
    try:
        value = ast.literal_eval(s)

        if not isinstance(value, list):
            raise ValueError("Not a list")

        return value

    except (ValueError, SyntaxError):
        return None   # or raise custom error