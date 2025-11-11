import os
import json
from typing import Annotated, Dict, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.llms import Ollama
from langsmith import Client
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
except Exception:  # noqa: PIE786
    Langfuse = None  # type: ignore
    CallbackHandler = None  # type: ignore

from .config import settings
from .prompts import (
    SYSTEM_PROMPTS,
    build_discovery_human_prompt,
    build_file_analysis_prompt,
    build_project_analysis_prompt,
    build_teaching_prompt,
    build_codebase_query_prompt,
    PROMPTS,
)
from .tools import tavily_search
from .ai_validator import AIValidator
from .token_utils import TokenTracker
from .py_agent import (
    extract_path_from_text,
    discover_codebase_files,
    build_file_structure_text,
    read_key_files_snippet,
    read_file_for_llm,
    build_project_context,
    build_query_context,
    read_content_for_parsing,
    get_last_user_text,
    determine_route,
)


# Inline IntentClassifier
class IntentClassifier:
    def __init__(self) -> None:
        self.classification_prompt = PROMPTS["intent_classification"]

    def classify_intent(self, content: str, file_path: str = "", session_id: str = "default") -> Dict[str, Any]:
        try:
            analysis_content = content[:5000]
            prompt = (
                f"{self.classification_prompt}\n\n"
                f"File Path: {file_path}\n"
                f"Content Preview:\n{analysis_content}\n\n"
                "Analyze this content and provide your classification."
            )
            with TokenTracker(
                operation="classify_intent",
                session_id=session_id,
                prompt=prompt,
                content=analysis_content,
                file_path=file_path
            ) as tracker:
                tracker.mark_generation_start()
                response = llm.invoke(prompt)
                result = str(response)
                tracker.mark_generation_done(result)
            return self._parse_classification_response(result)
        except Exception:
            return self._fallback_classification(content, file_path)

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        import json, re
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._fallback_classification("", "")
        except Exception:
            return self._fallback_classification("", "")

    def _fallback_classification(self, content: str, file_path: str) -> Dict[str, Any]:
        if file_path.endswith('.py'):
            return {"file_type": "code_file", "domain": "programming", "intent": "code_implementation", "relevant_entities": ["code_structure", "dependencies", "functions"], "extraction_focus": "Code structure and functionality", "confidence": 0.6}
        if 'import' in content.lower() and 'def ' in content:
            return {"file_type": "code_file", "domain": "programming", "intent": "code_implementation", "relevant_entities": ["code_structure", "dependencies", "functions"], "extraction_focus": "Code structure and functionality", "confidence": 0.7}
        return {"file_type": "document", "domain": "general", "intent": "information", "relevant_entities": ["general_info", "key_points"], "extraction_focus": "Key information and insights", "confidence": 0.5}


# Inline DynamicExtractor
class DynamicExtractor:
    def __init__(self) -> None:
        self.intent_classifier = IntentClassifier()
        self.extraction_prompt = PROMPTS["dynamic_extraction"]

    def extract_entities(self, content: str, file_path: str = "", session_id: str = "default") -> Dict[str, Any]:
        try:
            classification = self.intent_classifier.classify_intent(content, file_path, session_id=session_id)
            extraction_prompt = self._build_dynamic_prompt(content, classification)
            
            content_truncated = content[:8000]
            with TokenTracker(
                operation="extract_entities",
                session_id=session_id,
                prompt=extraction_prompt,
                content=content_truncated,
                file_path=file_path
            ) as tracker:
                tracker.mark_generation_start()
                if classification.get("file_type") in ["code_file", "notebook"]:
                    response = code_llm.invoke(extraction_prompt)
                else:
                    response = llm.invoke(extraction_prompt)
                result = str(response)
                tracker.mark_generation_done(result)
            
            extracted_data = self._parse_extraction_response(result)
            extracted_data["classification"] = classification
            return extracted_data
        except Exception as e:
            return self._fallback_extraction(content, file_path, str(e))

    def _build_dynamic_prompt(self, content: str, classification: Dict[str, Any]) -> str:
        base_prompt = self.extraction_prompt
        specific = (
            f"\n\nCONTENT ANALYSIS:\n"
            f"- File Type: {classification.get('file_type', 'unknown')}\n"
            f"- Domain: {classification.get('domain', 'unknown')}\n"
            f"- Intent: {classification.get('intent', 'unknown')}\n"
            f"- Focus: {classification.get('extraction_focus', 'general information')}\n"
            f"- Relevant Entities: {', '.join(classification.get('relevant_entities', []))}\n\n"
            f"EXTRACTION INSTRUCTIONS:\n"
            f"Focus on extracting information relevant to: {classification.get('extraction_focus', 'general information')}\n"
            f"Pay special attention to: {', '.join(classification.get('relevant_entities', []))}\n\n"
            f"CONTENT TO ANALYZE:\n{content[:8000]}\n\n"
            f"Extract all relevant information and return structured JSON."
        )
        return base_prompt + specific

    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        import json, re
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"general_info": {"purpose": "Information extraction", "key_points": [response[:500]], "extraction_notes": "Raw extraction without JSON structure"}, "extraction_quality": {"confidence": 0.5, "completeness": "low", "missing_info": ["structured_data"], "extraction_notes": "Could not parse structured JSON"}}
        except Exception as e:
            return self._fallback_extraction("", "", f"JSON parsing failed: {str(e)}")

    def _fallback_extraction(self, content: str, file_path: str, error: str) -> Dict[str, Any]:
        return {"general_info": {"purpose": "Error in extraction", "key_points": [f"Extraction failed: {error}"], "extraction_notes": "Fallback extraction due to error"}, "extraction_quality": {"confidence": 0.0, "completeness": "low", "missing_info": ["all_structured_data"], "extraction_notes": f"Extraction failed: {error}"}, "classification": {"file_type": "unknown", "domain": "unknown", "intent": "error", "confidence": 0.0}}


llm = Ollama(model=settings.ollama_general_model, base_url=settings.ollama_base_url, temperature=0.2)
code_llm = Ollama(model=settings.ollama_code_model, base_url=settings.ollama_base_url, temperature=0.1)
langsmith_client = Client(api_key=settings.langsmith_api_key) if settings.langsmith_api_key else Client()
langfuse = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_host) if (Langfuse and settings.langfuse_public_key and settings.langfuse_secret_key) else None
callback = CallbackHandler(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_host) if (CallbackHandler and settings.langfuse_public_key and settings.langfuse_secret_key) else None


class ProjectDetails(BaseModel):
    path: Optional[str] = None
    name: Optional[str] = None
    type: str = "none"
    summary: Optional[str] = None
    key_components: List[str] = Field(default_factory=list)
    main_entry_points: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    architecture_patterns: List[str] = Field(default_factory=list)
    discovered_files: List[Dict[str, Any]] = Field(default_factory=list)
    parsed_files: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[Dict[str, Any]] = None
    last_analyzed: Optional[str] = None


class ChatInfo(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_context: Optional[str] = None
    session_id: Optional[str] = None


class State(TypedDict):
    messages: Annotated[list, add_messages]
    chat_info: ChatInfo
    project_details: ProjectDetails
    context_vars: Dict[str, Any]
    classification: Dict[str, Any]


def codebase_discovery_agent(state: State) -> State:
    messages = state["messages"]
    project_details = state.get("project_details", ProjectDetails())
    user_text = get_last_user_text(messages)
    codebase_path = extract_path_from_text(str(user_text))
    if not codebase_path:
        return {"messages": [AIMessage(content="Please provide a codebase path after keywords like 'codebase', 'project', or 'file'.")]}
    codebase_path = codebase_path.strip().strip("'\"")
    if not os.path.exists(codebase_path):
        return {"messages": [AIMessage(content=f"Codebase path not found: {codebase_path}")]}
    discovered_files = discover_codebase_files(codebase_path)
    file_structure = build_file_structure_text(codebase_path, discovered_files)
    key_files_content = read_key_files_snippet(discovered_files)
    system_msg = SystemMessage(content=SYSTEM_PROMPTS["codebase_discovery"])
    human_msg = HumanMessage(content=build_discovery_human_prompt(file_structure, key_files_content))
    response = code_llm.invoke([system_msg, human_msg], config={"callbacks": [callback]} if callback else None)
    discovery_result = str(response)
    updated_project = ProjectDetails(path=codebase_path, type="codebase", discovered_files=discovered_files, summary=discovery_result, last_analyzed=str(os.path.getmtime(codebase_path) if os.path.exists(codebase_path) else ""))
    return {"messages": [AIMessage(content=discovery_result)], "project_details": updated_project}


def file_parsing_agent(state: State) -> State:
    discovered_files = state.get("discovered_files", [])
    if not discovered_files:
        return {"messages": [AIMessage(content="No files discovered yet. Please run codebase discovery first.")]}
    parsed_files = []
    for fi in discovered_files[:10]:
        file_path = fi["path"]
        try:
            content = read_content_for_parsing(fi)
            prompt = build_file_analysis_prompt(fi['relative_path'], fi['type'], content[:8000])
            response = code_llm.invoke(prompt, config={"callbacks": [callback]} if callback else None)
            analysis = str(response)
            parsed_files.append({"path": file_path, "relative_path": fi["relative_path"], "type": fi["type"], "analysis": analysis, "content_preview": content[:2000]})
        except Exception as e:
            parsed_files.append({"path": file_path, "relative_path": fi["relative_path"], "type": fi["type"], "analysis": f"Error parsing file: {e}", "content_preview": ""})
    return {"messages": [AIMessage(content=f"Parsed {len(parsed_files)} files successfully.")], "parsed_files": parsed_files}


def project_analysis_agent(state: State) -> State:
    parsed_files = state.get("parsed_files", [])
    codebase_path = state.get("codebase_path", "")
    if not parsed_files:
        return {"messages": [AIMessage(content="No parsed files available. Please run file parsing first.")]}
    project_context = build_project_context(parsed_files, codebase_path)
    prompt = build_project_analysis_prompt(project_context)
    response = code_llm.invoke(prompt, config={"callbacks": [callback]} if callback else None)
    project_analysis = str(response)
    return {"messages": [AIMessage(content=project_analysis)], "project_analysis": {"analysis": project_analysis, "files_analyzed": len(parsed_files)}, "project_summary": project_analysis, "key_components": [], "main_entry_points": [], "dependencies": [], "architecture_patterns": [], "analysis_context": f"Complete project analysis for {codebase_path}"}


def project_teaching_agent(state: State) -> State:
    project_analysis = state.get("project_analysis", {})
    parsed_files = state.get("parsed_files", [])
    codebase_path = state.get("codebase_path", "")
    if not project_analysis:
        return {"messages": [AIMessage(content="No project analysis available. Please analyze a project first.")]}
    key_file_lines = ""
    for fi in parsed_files[:10]:
        key_file_lines += f"\n- {fi['relative_path']}: {fi['analysis'][:300]}"
    prompt = build_teaching_prompt(codebase_path, len(parsed_files), project_analysis.get('analysis', 'No analysis available'), key_file_lines)
    response = code_llm.invoke(prompt, config={"callbacks": [callback]} if callback else None)
    teaching_content = str(response)
    return {"messages": [AIMessage(content=teaching_content)], "project_analysis": {**project_analysis, "teaching_guide": teaching_content}}


def chat_node(state: State) -> State:
    messages = state["messages"]
    # Inject dynamic context variables into the system prompt so LLM always sees them
    ctx_vars = state.get("context_vars", {}) or {}
    if ctx_vars:
        ctx_lines = []
        for k, v in list(ctx_vars.items())[:50]:
            try:
                ctx_lines.append(f"{k}: {v}")
            except Exception:
                continue
        ctx_block = "\n".join(ctx_lines)
        sys_text = SYSTEM_PROMPTS["general_chat"] + "\n\nContext Variables (authoritative facts extracted from files/chats):\n" + ctx_block
    else:
        sys_text = SYSTEM_PROMPTS["general_chat"]
    system_msg = SystemMessage(content=sys_text)
    last_user = next((m for m in reversed(messages) if getattr(m, "type", None) == "human"), None)
    if last_user:
        user_msg = HumanMessage(content=getattr(last_user, "content", ""))
        response = llm.invoke([system_msg, user_msg], config={"callbacks": [callback]} if callback else None)
    else:
        response = llm.invoke([system_msg] + messages, config={"callbacks": [callback]} if callback else None)
    return {"messages": [AIMessage(content=str(response))]}


def _flatten_extraction(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten commonly used fields into a dict for LLM context access."""
    flat: Dict[str, Any] = {}
    if not isinstance(extracted, dict):
        return flat
    # copy top-level obvious metrics
    sections = [
        "model_info", "training_info", "performance_metrics", "framework_info",
        "dataset_info", "code_structure", "research_info", "general_info"
    ]
    for sec in sections:
        data = extracted.get(sec)
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{sec}.{k}"
                flat[key] = v
    # Also add a few friendly aliases for frequent questions
    perf = extracted.get("performance_metrics") or {}
    train = extracted.get("training_info") or {}
    model = extracted.get("model_info") or {}
    if isinstance(perf, dict):
        if "accuracy" in perf: flat["accuracy"] = perf.get("accuracy")
        if "loss" in perf: flat["loss"] = perf.get("loss")
    if isinstance(train, dict):
        if "optimizer" in train: flat["optimizer"] = train.get("optimizer")
        if "loss_function" in train: flat["loss_function"] = train.get("loss_function")
        if "epochs" in train: flat["epochs"] = train.get("epochs")
        if "learning_rate" in train: flat["learning_rate"] = train.get("learning_rate")
    if isinstance(model, dict):
        if "type" in model: flat["model_type"] = model.get("type")
        if "parameters" in model: flat["parameters"] = model.get("parameters")
    return flat


def dynamic_context_node(state: State) -> State:
    """Analyze the latest user turn, extract structured info, and store as context_vars in state."""
    messages = state["messages"]
    last_user = None
    for m in reversed(messages):
        if getattr(m, "type", None) == "human" or (isinstance(m, dict) and (m.get("role") == "user")):
            last_user = m
            break
    user_text = getattr(last_user, "content", None) if last_user is not None else None
    if user_text is None and isinstance(last_user, dict):
        user_text = last_user.get("content")
    user_text = str(user_text or "")

    chat_info = state.get("chat_info", ChatInfo())
    if isinstance(chat_info, dict):
        session_id = chat_info.get("session_id", "default")
    else:
        session_id = getattr(chat_info, "session_id", "default") or "default"
    extractor = DynamicExtractor()
    extracted = extractor.extract_entities(user_text, "", session_id=session_id)
    flat = _flatten_extraction(extracted)
    cls = extracted.get("classification") if isinstance(extracted, dict) else {}
    return {"context_vars": flat, "classification": cls, "messages": []}


def codebase_query_agent(state: State) -> State:
    messages = state["messages"]
    project_analysis = state.get("project_analysis", {})
    parsed_files = state.get("parsed_files", [])
    codebase_path = state.get("codebase_path", "")
    project_summary = state.get("project_summary", "")
    key_components = state.get("key_components", [])
    main_entry_points = state.get("main_entry_points", [])
    dependencies = state.get("dependencies", [])
    architecture_patterns = state.get("architecture_patterns", [])
    analysis_context = state.get("analysis_context", "")
    if not project_analysis:
        return {"messages": [AIMessage(content="No codebase analysis available. Please analyze a codebase first using 'codebase <path>' or 'project <path>'.")]}
    user_question = get_last_user_text(messages)
    if isinstance(user_question, List):
        user_question = "\n".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in user_question])
    context = build_query_context(project_analysis, parsed_files, codebase_path, project_summary, key_components, main_entry_points, dependencies, architecture_patterns, analysis_context)
    prompt = build_codebase_query_prompt(context, user_question)
    response = llm.invoke(prompt, config={"callbacks": [callback]} if callback else None)
    return {"messages": [AIMessage(content=str(response))]}


def route_messages(state: State) -> str:
    # If the latest user turn appears to ask about technical details, populate dynamic context first.
    last_text = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or (isinstance(m, dict) and (m.get("role") == "user")):
            if isinstance(m, dict):
                last_text = m.get("content", "")
            else:
                last_text = getattr(m, "content", "")
            break
    lt = str(last_text or "").lower()
    tech_triggers = ["loss", "optimizer", "accuracy", "f1", "precision", "recall", "model", "dataset", ".py", ".ipynb"]
    if any(t in lt for t in tech_triggers):
        return "dynamic_context"
    return "chat"


def single_file_agent(state: State) -> State:
    messages = state["messages"]
    user_text = get_last_user_text(messages)
    file_path = extract_path_from_text(str(user_text))
    if not file_path:
        return {"messages": [AIMessage(content="Please provide a file path after the keyword 'file'.")]}
    file_path = file_path.strip().strip("'\"")
    if not os.path.exists(file_path):
        return {"messages": [AIMessage(content=f"File not found: {file_path}")]}
    file_blob = read_file_for_llm(file_path)
    if file_blob.get("kind") == "error":
        return {"messages": [AIMessage(content=file_blob.get("error", "Unknown file error"))]}
    prompt = (
        "You are an AI agent collaborating in a multi-turn chat.\n"
        "Task: 1) Answer the user's question naturally. 2) If a file is provided, summarize it and extract key details, likely outputs, and mention any images/plots if present.\n"
        "Keep the answer concise and helpful.\n\n"
        f"User message:\n{user_text}\n\n"
        f"File path: {file_path}\n"
        f"File content (truncated):\n{file_blob.get('text','')}\n"
        + ("\nNote: The notebook appears to contain images/plots." if file_blob.get("has_images") else "")
    )
    response = code_llm.invoke(prompt, config={"callbacks": [callback]} if callback else None)
    file_analysis = str(response)
    return {"messages": [AIMessage(content=file_analysis)], "current_file_path": file_path, "project_type": "single_file", "current_file_analysis": {"path": file_path, "analysis": file_analysis, "content_preview": file_blob.get('text', '')[:1000], "file_type": os.path.splitext(file_path)[1].lower()}, "analysis_context": f"Single file analysis for {file_path}"}


def build_graph() -> Any:
    graph_builder = StateGraph(State)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_node("dynamic_context", dynamic_context_node)
    graph_builder.add_node("discovery", codebase_discovery_agent)
    graph_builder.add_node("parsing", file_parsing_agent)
    graph_builder.add_node("analysis", project_analysis_agent)
    graph_builder.add_node("teaching", project_teaching_agent)
    graph_builder.add_node("codebase_query", codebase_query_agent)
    graph_builder.add_node("single_file", single_file_agent)
    graph_builder.add_conditional_edges(START, route_messages, {"chat": "chat", "dynamic_context": "dynamic_context", "discovery": "discovery", "parsing": "parsing", "analysis": "analysis", "teaching": "teaching", "codebase_query": "codebase_query", "single_file": "single_file", "quit": END})
    graph_builder.add_edge("chat", END)
    graph_builder.add_edge("dynamic_context", "chat")
    graph_builder.add_edge("discovery", END)
    graph_builder.add_edge("parsing", END)
    graph_builder.add_edge("analysis", END)
    graph_builder.add_edge("teaching", END)
    graph_builder.add_edge("codebase_query", END)
    graph_builder.add_edge("single_file", END)
    return graph_builder.compile()


# Inline SuggestionAgent
class SuggestionAgent:
    def build_grounded_suggestions(self, extracted_data: Dict[str, Any], use_web: bool = True, session_id: str = "default") -> Dict[str, Any]:
        context_lines: List[str] = []
        classification = extracted_data.get("classification", {})
        for section, data in (extracted_data or {}).items():
            if section == "classification":
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    if v is not None and v != "" and v != []:
                        context_lines.append(f"{section}.{k}: {v}")
        compact_context = "\n".join(context_lines[:80])

        web_snippets: List[str] = []
        citations: List[Dict[str, Any]] = []
        reference_content = ""
        if use_web:
            query_parts: List[str] = []
            if classification.get("intent"):
                query_parts.append(str(classification.get("intent")))
            if classification.get("domain"):
                query_parts.append(str(classification.get("domain")))
            mi = extracted_data.get("model_info", {})
            fw = extracted_data.get("framework_info", {})
            if isinstance(mi, dict) and mi.get("type"):
                query_parts.append(str(mi.get("type")))
            if isinstance(fw, dict) and fw.get("framework"):
                query_parts.append(str(fw.get("framework")))
            query = " ".join(query_parts) or "machine learning model improvement best practices"
            results = tavily_search(query, max_results=5)
            for r in results or []:
                title = r.get("title") or r.get("name") or ""
                url = r.get("url") or r.get("link") or ""
                snippet = r.get("snippet") or r.get("summary") or r.get("content") or ""
                if snippet:
                    web_snippets.append(f"- {title}: {snippet} (src: {url})")
                citations.append({"title": title, "url": url})
            reference_content = "\n".join(web_snippets)

        prompt = (
            PROMPTS["improvement_suggestions"] + "\n\n"
            "EXTRACTED FACTS:\n" + compact_context + "\n\n"
            "WEB EVIDENCE (optional):\n" + ("\n".join(web_snippets) if web_snippets else "(none)") + "\n\n"
            "Return ONLY the numbered list."
        )

        use_code_llm = classification.get("file_type") in ["code_file", "notebook"]
        
        with TokenTracker(
            operation="build_grounded_suggestions",
            session_id=session_id,
            prompt=prompt,
            content=compact_context,
            reference_content=reference_content
        ) as tracker:
            tracker.mark_generation_start()
            raw = str((code_llm if use_code_llm else llm).invoke(prompt))
            tracker.mark_generation_done(raw)
        
        suggestions = [s.strip() for s in raw.splitlines() if s.strip() and (s.strip()[0].isdigit() or s.strip().startswith("-"))]
        if not suggestions:
            import re as _re
            sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
            suggestions = [f"{i+1}. {s}" for i, s in enumerate(sentences[:10])]

        return {"suggestions": suggestions[:12], "citations": citations}


