import os
import re
import json
import glob
from typing import Dict, Any, List, Tuple
import logging


def extract_path_from_text(text: str) -> str:
    patterns = [r"[A-Za-z]:\\[^\s\"]+", r"\./[^\s\"]+", r"\../[^\s\"]+", r"/[^\s\"]+"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    m2 = re.search(r"(?:file|codebase|project)\s*[:=]\s*([\S]+)", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    return ""


def discover_codebase_files(codebase_path: str) -> List[Dict[str, Any]]:
    py_files = glob.glob(os.path.join(codebase_path, "**/*.py"), recursive=True)
    ipynb_files = glob.glob(os.path.join(codebase_path, "**/*.ipynb"), recursive=True)
    config_files = []
    for pattern in ["requirements.txt", "setup.py", "pyproject.toml", "environment.yml", "Dockerfile", "README.md"]:
        config_files.extend(glob.glob(os.path.join(codebase_path, f"**/{pattern}"), recursive=True))
    discovered = []
    for file_path in py_files + ipynb_files + config_files:
        try:
            rel_path = os.path.relpath(file_path, codebase_path)
            file_size = os.path.getsize(file_path)
            discovered.append({"path": file_path, "relative_path": rel_path, "type": "python" if file_path.endswith(".py") else "notebook" if file_path.endswith(".ipynb") else "config", "size": file_size})
        except Exception:
            continue
    return discovered


def build_file_structure_text(codebase_path: str, discovered_files: List[Dict[str, Any]]) -> str:
    file_structure = f"Codebase: {codebase_path}\n\nDiscovered Files:\n"
    for fi in discovered_files:
        file_structure += f"- {fi['relative_path']} ({fi['type']}, {fi['size']} bytes)\n"
    return file_structure


def read_key_files_snippet(discovered_files: List[Dict[str, Any]], limit: int = 5, max_chars: int = 2000) -> str:
    out = ""
    for fi in discovered_files[:limit]:
        try:
            if fi["type"] == "python":
                with open(fi["path"], "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    out += f"\n--- {fi['relative_path']} ---\n{content[:max_chars]}\n"
            elif fi["type"] == "notebook":
                with open(fi["path"], "r", encoding="utf-8", errors="ignore") as f:
                    nb = json.load(f)
                    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
                    code_content = "\n".join("".join(c.get("source", [])) for c in code_cells)
                    out += f"\n--- {fi['relative_path']} ---\n{code_content[:max_chars]}\n"
        except Exception as e:
            out += f"\n--- {fi['relative_path']} ---\nError reading file: {e}\n"
    return out


def read_file_for_llm(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ipynb":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                nb = json.load(f)
        except Exception as e:
            return {"kind": "error", "error": f"Failed to read notebook: {e}"}
        cells = nb.get("cells", [])
        md_texts: List[str] = []
        code_texts: List[str] = []
        has_images = False
        for c in cells:
            if c.get("cell_type") == "markdown":
                md_texts.append("".join(c.get("source", [])))
            elif c.get("cell_type") == "code":
                code_texts.append("".join(c.get("source", [])))
                for out in c.get("outputs", []) or []:
                    data = out.get("data") or {}
                    if isinstance(data, dict) and ("image/png" in data or "image/jpeg" in data):
                        has_images = True
        combined = "# Notebook Markdown (truncated)\n" + "\n\n".join(md_texts)[:6000] + "\n\n# Notebook Code (truncated)\n" + "\n\n".join(code_texts)[:12000]
        return {"kind": "ipynb", "text": combined, "has_images": has_images}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return {"kind": "text", "text": content[:20000]}
    except Exception as e:
        logging.exception("Failed to read file")
        return {"kind": "error", "error": f"Failed to read file: {e}"}


def build_project_context(parsed_files: List[Dict[str, Any]], codebase_path: str) -> str:
    ctx = f"""
    Analyze this entire codebase project and provide comprehensive insights:

    Project Path: {codebase_path}
    Total Files Analyzed: {len(parsed_files)}

    File Analysis Summary:
    """
    for fi in parsed_files:
        ctx += f"\n--- {fi['relative_path']} ({fi['type']}) ---\n"
        ctx += fi['analysis'][:1000] + "\n"
    return ctx


def build_query_context(project_analysis: Dict[str, Any], parsed_files: List[Dict[str, Any]], codebase_path: str, project_summary: str, key_components: List[str], main_entry_points: List[str], dependencies: List[str], architecture_patterns: List[str], analysis_context: str) -> str:
    ctx = f"""
    You are a codebase expert answering questions about this analyzed project.

    Project Context: {analysis_context}
    Project Path: {codebase_path}
    Files Analyzed: {len(parsed_files)}

    Project Summary:
    {project_summary}

    Key Components: {', '.join(key_components) if key_components else 'Not extracted yet'}
    Main Entry Points: {', '.join(main_entry_points) if main_entry_points else 'Not extracted yet'}
    Dependencies: {', '.join(dependencies) if dependencies else 'Not extracted yet'}
    Architecture Patterns: {', '.join(architecture_patterns) if architecture_patterns else 'Not extracted yet'}

    Detailed Analysis:
    {project_analysis.get('analysis', 'No analysis available')}

    Available Files:
    """
    for fi in parsed_files[:5]:
        ctx += f"\n- {fi['relative_path']} ({fi['type']}): {fi['analysis'][:200]}"
    return ctx


def read_content_for_parsing(file_info: Dict[str, Any]) -> str:
    file_path = file_info["path"]
    if file_info["type"] == "python":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if file_info["type"] == "notebook":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        md_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "markdown"]
        return f"# Code cells:\n{chr(10).join(''.join(c.get('source', [])) for c in code_cells)}\n\n# Markdown cells:\n{chr(10).join(''.join(c.get('source', [])) for c in md_cells)}"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_last_user_text(messages: List[Any]) -> str:
    last_user_text = None
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role == "user" or role == "human":
            last_user_text = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            break
    return str(last_user_text or "")


def determine_route(messages: List[Any], project_details: Dict[str, Any]) -> str:
    text = get_last_user_text(messages).lower()
    # Prefer direct path detection for single-file requests
    raw_text = get_last_user_text(messages)
    if extract_path_from_text(raw_text):
        logging.debug("Routing to single_file (path detected)")
        return "single_file"
    if "quit" in text or "exit" in text:
        return "quit"
    if any(keyword in text for keyword in ["codebase", "project", "analyze"]):
        if "discover" in text or "scan" in text:
            return "discovery"
        if "parse" in text or "files" in text:
            return "parsing"
        if "analyze" in text or "analysis" in text:
            return "analysis"
        if "teach" in text or "learn" in text or "recreate" in text or "how to" in text:
            return "teaching"
        return "discovery"
    pd_type = getattr(project_details, "type", None) if not isinstance(project_details, dict) else project_details.get("type")
    pd_analysis = getattr(project_details, "analysis", None) if not isinstance(project_details, dict) else project_details.get("analysis")
    if pd_type == "codebase" and pd_analysis and any(k in text for k in ["how", "what", "where", "why", "explain", "show", "tell", "analyze", "understand"]):
        logging.debug("Routing to codebase_query")
        return "codebase_query"
    if pd_type == "single_file" and pd_analysis and any(k in text for k in ["how", "what", "where", "why", "explain", "show", "tell"]):
        logging.debug("Routing to codebase_query (single file)")
        return "codebase_query"
    if " file " in f" {text} " or text.startswith("file ") or text.endswith(" file") or text == "file":
        logging.debug("Routing to single_file")
        return "single_file"
    logging.debug("Routing to chat")
    return "chat"


