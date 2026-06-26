"""
Pure-function tool dispatch for both ReactAgent and PlannerR1Agent.

execute_tool() is lifted from the per-action branches in ReactAgent.step() and
accepts a dict of arguments (as returned by native tool-calling) rather than a
raw comma-split string.  It returns (observation_str, new_current_data, is_terminal).
"""

import re
from pandas import DataFrame


class DateError(Exception):
    pass


def validate_date_format(date_str: str) -> bool:
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise DateError
    return True


def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not a valid city in the dataset.")
    return True


def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)


def execute_tool(name: str, args: dict, tools: dict, city_set: list,
                 retry_record: dict, current_data) -> tuple:
    """
    Dispatch a single tool call.

    Parameters
    ----------
    name          : tool name matching an entry in TOOL_SCHEMAS / actionMapping
    args          : dict of arguments from native tool-calling (already parsed JSON)
    tools         : dict of instantiated tool objects (from ReactAgent.load_tools)
    city_set      : list of valid city names
    retry_record  : mutable retry counter dict (updated in place on failure)
    current_data  : most recently retrieved DataFrame or other data

    Returns
    -------
    (observation_str, new_current_data, is_terminal)
    is_terminal is True only for the Planner tool.
    """

    # helper: mask the previous large data block to keep context short
    def _mask(scratchpad_unused, data):
        return data  # masking is scratchpad-specific; callers handle it if needed

    if name == 'FlightSearch':
        dep  = args.get('departure_city', '')
        dest = args.get('destination_city', '')
        date = args.get('date', '')
        try:
            validate_date_format(date)
            validate_city_format(dep, city_set)
            validate_city_format(dest, city_set)
            new_data = tools['flights'].run(dep, dest, date)
            obs = to_string(new_data)
            retry_record = _reset_record(retry_record)
            return obs, new_data, False
        except DateError:
            retry_record['flights'] = retry_record.get('flights', 0) + 1
            obs = f"'{date}' is not in the format YYYY-MM-DD"
            return obs, current_data, False
        except ValueError as e:
            retry_record['flights'] = retry_record.get('flights', 0) + 1
            return str(e), current_data, False
        except Exception as e:
            retry_record['flights'] = retry_record.get('flights', 0) + 1
            return 'Illegal Flight Search. Please try again.', current_data, False

    elif name == 'AccommodationSearch':
        city = args.get('city', '')
        try:
            validate_city_format(city, city_set)
            new_data = tools['accommodations'].run(city)
            obs = to_string(new_data).strip('\n').strip()
            retry_record = _reset_record(retry_record)
            return obs, new_data, False
        except ValueError as e:
            retry_record['accommodations'] = retry_record.get('accommodations', 0) + 1
            return str(e), current_data, False
        except Exception as e:
            retry_record['accommodations'] = retry_record.get('accommodations', 0) + 1
            return 'Illegal Accommodation Search. Please try again.', current_data, False

    elif name == 'RestaurantSearch':
        city = args.get('city', '')
        try:
            validate_city_format(city, city_set)
            new_data = tools['restaurants'].run(city)
            obs = to_string(new_data).strip()
            retry_record = _reset_record(retry_record)
            return obs, new_data, False
        except ValueError as e:
            retry_record['restaurants'] = retry_record.get('restaurants', 0) + 1
            return str(e), current_data, False
        except Exception as e:
            retry_record['restaurants'] = retry_record.get('restaurants', 0) + 1
            return 'Illegal Restaurant Search. Please try again.', current_data, False

    elif name == 'AttractionSearch':
        city = args.get('city', '')
        try:
            validate_city_format(city, city_set)
            new_data = tools['attractions'].run(city)
            obs = to_string(new_data).strip('\n').strip()
            retry_record = _reset_record(retry_record)
            return obs, new_data, False
        except ValueError as e:
            retry_record['attractions'] = retry_record.get('attractions', 0) + 1
            return str(e), current_data, False
        except Exception as e:
            retry_record['attractions'] = retry_record.get('attractions', 0) + 1
            return 'Illegal Attraction Search. Please try again.', current_data, False

    elif name == 'CitySearch':
        state = args.get('state', '')
        try:
            obs = to_string(tools['cities'].run(state)).strip()
            retry_record = _reset_record(retry_record)
            return obs, current_data, False
        except ValueError as e:
            retry_record['cities'] = retry_record.get('cities', 0) + 1
            return str(e), current_data, False
        except Exception as e:
            retry_record['cities'] = retry_record.get('cities', 0) + 1
            return 'Illegal City Search. Please try again.', current_data, False

    elif name == 'GoogleDistanceMatrix':
        origin = args.get('origin', '')
        dest   = args.get('destination', '')
        mode   = args.get('mode', '')
        try:
            new_data = tools['googleDistanceMatrix'].run(origin, dest, mode)
            obs = to_string(new_data)
            retry_record = _reset_record(retry_record)
            return obs, new_data, False
        except Exception as e:
            retry_record['googleDistanceMatrix'] = retry_record.get('googleDistanceMatrix', 0) + 1
            return 'Illegal GoogleDistanceMatrix. Please try again.', current_data, False

    elif name == 'NotebookWrite':
        description = args.get('description', '')
        try:
            obs = str(tools['notebook'].write(current_data, description))
            retry_record = _reset_record(retry_record)
            return obs, current_data, False
        except Exception as e:
            retry_record['notebook'] = retry_record.get('notebook', 0) + 1
            return str(e), current_data, False

    elif name == 'Planner':
        query = args.get('query', '') or args.get('Query', '')
        try:
            obs = str(tools['planner'].run(str(tools['notebook'].list_all()), query))
            retry_record = _reset_record(retry_record)
            return obs, current_data, True  # is_terminal=True
        except Exception as e:
            retry_record['planner'] = retry_record.get('planner', 0) + 1
            return str(e), current_data, False

    else:
        retry_record['invalidAction'] = retry_record.get('invalidAction', 0) + 1
        valid = ('FlightSearch, AccommodationSearch, RestaurantSearch, AttractionSearch, '
                 'CitySearch, GoogleDistanceMatrix, NotebookWrite')
        return f"Unknown tool '{name}'. Valid tools: {valid}.", current_data, False


def _reset_record(retry_record: dict) -> dict:
    """Reset all counters to 0 after a successful tool call."""
    return {k: 0 for k in retry_record}
