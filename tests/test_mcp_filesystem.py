from types import SimpleNamespace

import pytest

from app.config import MCPFilesystemConfig
from app.mcp_client import filesystem


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "inputSchema": {"type": "object", "properties": {}},
    }


def test_mcp_tool_allowlist_is_read_only_without_request_approval():
    discovered = [
        _tool("read_text_file"),
        _tool("write_file"),
        _tool("delete_file"),
    ]

    read_only = filesystem._openai_tools(discovered, allow_mutations=False)
    approved = filesystem._openai_tools(discovered, allow_mutations=True)

    assert [tool["function"]["name"] for tool in read_only] == ["read_text_file"]
    assert [tool["function"]["name"] for tool in approved] == [
        "read_text_file",
        "write_file",
    ]


def test_model_generated_path_cannot_escape_configured_root(tmp_path):
    filesystem._validate_paths({"path": str(tmp_path / "inside.txt")}, (tmp_path,))

    with pytest.raises(filesystem.FilesystemMCPError, match="outside configured"):
        filesystem._validate_paths(
            {"path": str(tmp_path.parent / "outside.txt")}, (tmp_path,)
        )


def test_filesystem_answer_runs_bounded_tool_loop(monkeypatch, tmp_path):
    config = MCPFilesystemConfig(
        roots=(tmp_path,),
        allow_mutations=False,
        max_tool_rounds=3,
        operation_timeout_seconds=5,
    )
    monkeypatch.setattr(
        filesystem,
        "get_settings",
        lambda: SimpleNamespace(mcp_filesystem=config),
    )

    async def fake_list_tools(_config):
        return [_tool("read_text_file"), _tool("write_file")]

    async def fake_call_tool(_config, name, arguments):
        assert name == "read_text_file"
        assert arguments["path"] == str(tmp_path / "README.md")
        return "project documentation", False

    monkeypatch.setattr(filesystem, "_list_tools_async", fake_list_tools)
    monkeypatch.setattr(filesystem, "_call_tool_async", fake_call_tool)

    class ToolLLM:
        supports_tool_calling = True
        provider_name = "azure_foundry"
        model_name = "Kimi-K2.6"

        def __init__(self):
            self.calls = 0

        def complete_with_tools(self, *, messages, tools, tool_choice):
            self.calls += 1
            assert [tool["function"]["name"] for tool in tools] == [
                "read_text_file"
            ]
            if self.calls == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_text_file",
                                "arguments": (
                                    '{"path":"' + str(tmp_path / "README.md").replace("\\", "\\\\") + '"}'
                                ),
                            },
                        }
                    ],
                }
            assert messages[-1]["role"] == "tool"
            return {"role": "assistant", "content": "The README was read."}

    response = filesystem.answer_filesystem_question(
        "Read README.md", llm_client=ToolLLM()
    )

    assert response.status == "ok"
    assert response.answer == "The README was read."
    assert len(response.tool_records) == 1
    assert response.tool_records[0].name == "read_text_file"


def test_managed_document_writer_is_confined_and_uses_mcp(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    managed = root / "knowledge" / "v1"
    managed.mkdir(parents=True)
    config = MCPFilesystemConfig(
        roots=(root,),
        allow_mutations=True,
        operation_timeout_seconds=5,
    )

    def fake_run(coroutine, _timeout):
        coroutine.close()
        return [
            filesystem.FilesystemToolRecord(
                name="write_file",
                arguments={"path": str(managed / "Table.txt")},
                result="written",
            )
        ]

    monkeypatch.setattr(filesystem, "_run_with_timeout", fake_run)
    records = filesystem.write_managed_text_files(
        {managed / "Table.txt": "table documentation"},
        managed_root=managed,
        config=config,
    )
    assert records[0].name == "write_file"

    with pytest.raises(filesystem.FilesystemMCPError, match="outside its output"):
        filesystem.write_managed_text_files(
            {root / "outside.txt": "blocked"},
            managed_root=managed,
            config=config,
        )
