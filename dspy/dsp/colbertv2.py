from typing import Any

import requests

from dspy.clients.cache import request_cache
from dspy.dsp.utils import dotdict

# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: str | int | None = None,
        post_requests: bool = False,
    ):
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self,
        query: str,
        k: int = 10,
        simplify: bool = False,
    ) -> list[str] | list[dotdict]:
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@request_cache()
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert k <= 100, "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)
    res.raise_for_status()

    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")

    topk = res_json["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


@request_cache()
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped


@request_cache()
def colbertv2_post_request_v2(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)
    res.raise_for_status()

    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")

    return res_json["topk"][:k]


@request_cache()
def colbertv2_post_request_v2_wrapped(*args, **kwargs):
    return colbertv2_post_request_v2(*args, **kwargs)


colbertv2_post_request = colbertv2_post_request_v2_wrapped
