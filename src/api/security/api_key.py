from fastapi import Header, HTTPException


API_KEY = "retail-enterprise-key"


def validate_api_key(
    x_api_key: str = Header(None),
):

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )

    return x_api_key
