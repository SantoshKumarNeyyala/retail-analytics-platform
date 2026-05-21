import redis
import json

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)


def get_cached_prediction(
    key: str,
):

    value = redis_client.get(key)

    if value:

        return json.loads(value)

    return None


def set_cached_prediction(
    key: str,
    value: dict,
    expiry: int = 3600,
):

    redis_client.setex(
        key,
        expiry,
        json.dumps(value),
    )
