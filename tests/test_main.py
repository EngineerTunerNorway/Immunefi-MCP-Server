import importlib.util
import json
import sys
import types
import unittest


MODULE_PATH = "/home/runner/work/Immunefi-MCP-Server/Immunefi-MCP-Server/main.py"


def load_main_module():
    httpx_mod = types.ModuleType("httpx")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    httpx_mod.AsyncClient = DummyClient
    sys.modules["httpx"] = httpx_mod

    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def deco(fn):
                return fn

            return deco

        def run(self, transport="stdio"):
            return None

    fastmcp_mod.FastMCP = FastMCP
    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.fastmcp"] = fastmcp_mod

    spec = importlib.util.spec_from_file_location("main", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardenedServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_module()

    def test_validate_ids_rejects_non_ascii(self):
        err = self.main._validate_ids(["ñ"])
        self.assertIn("invalid project id format", err)

    def test_max_bounty_handles_null(self):
        self.assertIsNone(self.main._max_bounty({"maxBounty": None}))

    def test_response_size_cap_returns_error(self):
        payload = {"body": "a" * (self.main.MAX_RESPONSE_BYTES + 1000)}
        out = json.loads(self.main._ok(payload))
        self.assertEqual(out["error"], "response_too_large")


if __name__ == "__main__":
    unittest.main()
