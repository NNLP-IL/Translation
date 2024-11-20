import json


def serialize(record):
    subset = {
        "timestamp": record["time"].timestamp(),
        "message": record["message"],
        "level": record["level"].name,
    }
    return json.dumps(subset)


def patching(record):
    record["extra"]["serialized"] = serialize(record)
