import pytest
import requests

from fpl_predictor.api import FPLAPIClient, FPLAPIError


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


def test_successful_response_parsing():
    session = FakeSession(FakeResponse({"elements": []}))
    client = FPLAPIClient(session=session, timeout=7)
    assert client.get_bootstrap_static() == {"elements": []}
    assert session.calls[0][1] == 7


def test_http_error_has_clear_custom_exception():
    response = FakeResponse(error=requests.HTTPError("503 Server Error"))
    client = FPLAPIClient(session=FakeSession(response))
    with pytest.raises(FPLAPIError, match="Failed to fetch valid FPL data"):
        client.get_fixtures()


def test_wrong_payload_shape_raises_custom_exception():
    client = FPLAPIClient(session=FakeSession(FakeResponse({"not": "fixtures"})))
    with pytest.raises(FPLAPIError, match="not a JSON list"):
        client.get_fixtures()
