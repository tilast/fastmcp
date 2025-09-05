"""Tests for stateless OAuth proxy functionality.

These tests verify that the OAuth proxy can survive pod restarts and handle
refresh token flows even when the in-memory client registry is cleared.
"""

import time
import uuid

import pytest
from mcp.server.auth.provider import AccessToken

from fastmcp.server.auth.auth import TokenVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy, ProxyDCRClient


class MockTokenVerifier(TokenVerifier):
    """Mock token verifier for testing."""

    def __init__(self):
        super().__init__(required_scopes=["openid", "email"])

    async def verify_token(self, token: str) -> AccessToken | None:
        """Mock token verification - always returns valid token for testing."""
        return AccessToken(
            token=token,
            client_id="test-client",
            scopes=self.required_scopes,
            expires_at=int(time.time() + 3600),
        )


def _minimal_proxy():
    """Construct a minimal OAuth proxy for testing."""
    return OAuthProxy(
        upstream_authorization_endpoint="https://upstream.example/authorize",
        upstream_token_endpoint="https://upstream.example/token",
        upstream_client_id="UP_CLIENT",
        upstream_client_secret="UP_SECRET",
        token_verifier=MockTokenVerifier(),
        base_url="https://mcp.example",
        redirect_path="/auth/callback",
        allowed_client_redirect_uris=[
            "http://localhost:*",
            "https://claude.ai/api/mcp/auth_callback",
        ],
    )


@pytest.mark.asyncio
async def test_get_client_reconstructs_after_restart():
    """
    If the pod restarted and the in-memory _clients map is empty,
    get_client() should reconstruct a ProxyDCRClient so refresh can proceed.
    """
    proxy = _minimal_proxy()
    missing_id = str(uuid.uuid4())

    # Map was cleared (simulating restart)
    proxy._clients.clear()

    # Should reconstruct the client
    client = await proxy.get_client(missing_id)
    assert client is not None
    assert client.client_id == missing_id

    # And it should persist for the next lookup
    assert missing_id in proxy._clients

    # Subsequent lookup should return the same client
    client2 = await proxy.get_client(missing_id)
    assert client2 is client


@pytest.mark.asyncio
async def test_reconstructed_client_has_correct_properties():
    """
    Verify that a reconstructed client has all the necessary properties
    for OAuth flows to work correctly.
    """
    proxy = _minimal_proxy()
    client_id = str(uuid.uuid4())

    # Clear clients to simulate restart
    proxy._clients.clear()

    # Reconstruct client
    client = await proxy.get_client(client_id)
    assert client is not None

    # Verify properties
    assert client.client_id == client_id
    assert client.client_secret is None  # No secret for DCR clients
    assert client.grant_types == ["authorization_code", "refresh_token"]
    assert client.token_endpoint_auth_method == "none"

    # Check that ProxyDCRClient has the allowed redirect patterns
    assert isinstance(client, ProxyDCRClient)
    assert client._allowed_redirect_uri_patterns == [
        "http://localhost:*",
        "https://claude.ai/api/mcp/auth_callback",
    ]


@pytest.mark.asyncio
async def test_get_client_returns_existing_if_present():
    """
    Verify that get_client returns the existing client if it's already
    in the registry, without reconstruction.
    """
    proxy = _minimal_proxy()

    # Register a client normally
    client_id = str(uuid.uuid4())
    original_client = ProxyDCRClient(
        client_id=client_id,
        client_secret="test-secret",
        client_id_issued_at=int(time.time()),
        client_secret_expires_at=0,
        redirect_uris=["http://localhost:8080/callback"],
        grant_types=["authorization_code"],
        token_endpoint_auth_method="client_secret_post",
        allowed_redirect_uri_patterns=["http://localhost:*"],
        scope="openid email profile",
    )
    proxy._clients[client_id] = original_client

    # Get should return the existing client
    fetched = await proxy.get_client(client_id)
    assert fetched is original_client
    assert fetched.client_secret == "test-secret"  # Should keep original properties


@pytest.mark.asyncio
async def test_multiple_clients_can_be_reconstructed():
    """
    Verify that multiple clients can be reconstructed independently
    after a restart.
    """
    proxy = _minimal_proxy()

    # Clear to simulate restart
    proxy._clients.clear()

    # Reconstruct multiple clients
    client_ids = [str(uuid.uuid4()) for _ in range(3)]
    clients = []

    for client_id in client_ids:
        client = await proxy.get_client(client_id)
        assert client is not None
        assert client.client_id == client_id
        clients.append(client)

    # All should be persisted
    assert len(proxy._clients) == 3
    for client_id in client_ids:
        assert client_id in proxy._clients

    # Each should be unique (check by id since ProxyDCRClient isn't hashable)
    assert len(set(id(c) for c in clients)) == 3


@pytest.mark.asyncio
async def test_reconstructed_client_scope_matches_proxy_requirements():
    """
    Verify that reconstructed clients get the correct scope from the proxy's
    required_scopes/default_scope_str.
    """
    proxy = _minimal_proxy()
    client_id = str(uuid.uuid4())

    # Clear clients
    proxy._clients.clear()

    # Reconstruct
    client = await proxy.get_client(client_id)
    assert client is not None

    # Should have the proxy's default scope
    expected_scope = " ".join(proxy.required_scopes or [])
    assert client.scope == expected_scope
