"""
OpenAI function-calling schemas for the TravelPlanner search tools.
NotebookWrite and Planner are intentionally excluded: in planner_r1 mode the
model keeps all observations in chat history (ToolMessages) and produces the
final itinerary directly, so neither tool is needed.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "FlightSearch",
            "description": "Search for flights between two cities on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure_city": {
                        "type": "string",
                        "description": "The city you will fly out from.",
                    },
                    "destination_city": {
                        "type": "string",
                        "description": "The city you aim to reach.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of travel in YYYY-MM-DD format.",
                    },
                },
                "required": ["departure_city", "destination_city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AccommodationSearch",
            "description": "Discover hotel rooms and accommodations available in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to search for accommodations.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "RestaurantSearch",
            "description": "Explore dining options in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to search for restaurants.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AttractionSearch",
            "description": "Find tourist attractions and points of interest in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to search for attractions.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CitySearch",
            "description": "List cities in a US state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "The US state to list cities for.",
                    },
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GoogleDistanceMatrix",
            "description": "Estimate the distance, travel time, and cost between two cities by self-driving or taxi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure city.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["self-driving", "taxi"],
                        "description": "Mode of transport.",
                    },
                },
                "required": ["origin", "destination", "mode"],
            },
        },
    },
]
