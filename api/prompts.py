SYSTEM_PROMPTS = {
    "codebase_discovery": """You are a codebase discovery agent. Analyze directory structures and identify project components, dependencies, and architecture patterns. Provide structured insights about codebases.""",
    "file_parsing": """You are a file parsing agent. Analyze individual code files and extract functionality, dependencies, patterns, and relationships. Focus on code structure and purpose.""",
    "project_analysis": """You are a senior software architect. Analyze complete projects, understand relationships between components, and provide comprehensive architectural insights.""",
    "project_teaching": """You are an expert software engineering instructor. Teach how to understand, recreate, and extend projects based on code analysis. Provide educational guidance.""",
    "codebase_query": """You are a codebase expert. Answer questions about analyzed projects using stored project information. Provide accurate, context-aware responses.""",
    "general_chat": """You are a helpful AI assistant. Provide general assistance and answer questions. Be concise and helpful.""",
}

def build_discovery_human_prompt(file_structure: str, key_files_content: str) -> str:
    return f"""
    Analyze this codebase structure and provide insights:

    {file_structure}

    Key Files Content (truncated):
    {key_files_content}

    Provide:
    1. Project purpose and main functionality
    2. Key components and their roles
    3. Entry points and main files
    4. Dependencies and relationships
    5. Architecture patterns observed
    6. Any notable features or patterns
    """


def build_file_analysis_prompt(relative_path: str, file_type: str, content: str) -> str:
    return f"""
    Analyze this file and provide:
    1. Purpose and functionality
    2. Key functions, classes, and imports
    3. Dependencies and relationships to other files
    4. Expected outputs or behaviors
    5. Any notable patterns or architectures

    File: {relative_path}
    Type: {file_type}

    Content:
    {content}
    """


def build_project_analysis_prompt(project_context: str) -> str:
    return f"""
    You are a senior software architect analyzing a complete codebase project.

    {project_context}

    Please provide a comprehensive project analysis including:

    1. **Project Overview**: What is this project about? What problem does it solve?
    2. **Architecture**: How is the codebase organized? What are the main components?
    3. **Dependencies**: How do files relate to each other? What are the import relationships?
    4. **Data Flow**: How does data flow through the system?
    5. **Key Features**: What are the main functionalities and capabilities?
    6. **Entry Points**: How do you run or use this project?
    7. **Dependencies**: What external libraries or frameworks are used?
    8. **Potential Issues**: Any obvious problems or areas for improvement?
    9. **Documentation**: What would you need to know to understand and use this project?
    """


def build_teaching_prompt(project_path: str, files_analyzed: int, analysis_text: str, key_file_lines: str) -> str:
    return f"""
    You are an expert software engineering instructor. Based on the following analyzed project, 
    teach someone how to understand, recreate, or extend this project.

    Project Path: {project_path}
    Files Analyzed: {files_analyzed}

    Project Analysis:
    {analysis_text}

    Key Files and Their Purposes:
    {key_file_lines}

    Please provide a comprehensive teaching guide that includes:

    1. **Project Understanding**: What this project does and why it's structured this way
    2. **Step-by-Step Recreation**: How to recreate this project from scratch
    3. **Key Concepts**: Important programming concepts, patterns, and techniques used
    4. **Dependencies Setup**: How to set up the environment and install dependencies
    5. **Architecture Decisions**: Why certain architectural choices were made
    6. **Best Practices**: What good practices this project demonstrates
    7. **Common Pitfalls**: What to avoid when creating similar projects
    8. **Extension Points**: How to add new features or modify existing ones
    9. **Testing Strategy**: How to test this type of project
    10. **Deployment Considerations**: How to deploy or distribute this project
    """


def build_codebase_query_prompt(context: str, user_question: str) -> str:
    return f"""
    {context}

    User Question: {user_question}

    Please answer the user's question about this codebase project. Use the stored project information to provide accurate, context-aware responses. 
    Be specific and reference actual files, functions, and code patterns. 
    If the question is about a specific file or functionality, provide detailed information based on the analysis.
    If you don't have enough information, say so clearly.
    """


PROMPTS = {
    "intent_classification": """
You are an expert AI agent that analyzes file content to understand its intent and purpose.
Analyze and return JSON with: file_type, domain, intent, relevant_entities[], extraction_focus, confidence (0-1).
""",
    "dynamic_extraction": """
You extract structured information based on content and prior classification. Return comprehensive JSON with sections like model_info, training_info, performance_metrics, framework_info, dataset_info, code_structure, research_info, general_info, extraction_quality.
Only include facts present in the content. Do not hallucinate.
""",
    "key_points_formatter": """
You are a precise summarizer. Create 6-10 crisp, standalone bullet points that capture the most important, non-redundant facts about the provided file. Prefer concrete values (model, optimizer, metrics, dataset, framework, functions/classes) over prose. Avoid paragraphs and avoid prefacing.

Extracted data (flattened):
{extracted_flat}

If useful, you may consult the original content (truncated):
{content}

Return ONLY bullet points, each on its own line, starting with '- '.
""",
    "improvement_suggestions": """
You are an expert reviewer. Based on extracted facts and optional web evidence, produce 6-12 practical, core improvements. Be specific and non-generic. Avoid hallucinations. Output a numbered list only.
""",
    "validation": """
Validate extracted info against original content. Return JSON with validation_results (is_accurate, confidence_score, completeness_score, hallucination_risk), plus details.
""",
}

