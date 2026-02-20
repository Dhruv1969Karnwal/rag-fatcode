"""
Utility functions for FastCode
"""

import os
import hashlib
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import tiktoken

# Global verbose logging flag
_VERBOSE_LOGGING = False


def set_verbose_logging(enabled: bool):
    """Enable or disable verbose logging output"""
    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = enabled


def is_verbose_logging() -> bool:
    """Check if verbose logging is enabled"""
    return _VERBOSE_LOGGING


def _print_box(title: str, content: List[str], width: int = 70, emoji: str = ""):
    """
    Print a formatted box with title and content.
    
    Args:
        title: Box title
        content: List of content lines
        width: Box width (default 70)
        emoji: Emoji to display in title
    """
    if not _VERBOSE_LOGGING:
        return
    
    # Build box
    border_char = "═"
    corner_tl = "╔"
    corner_tr = "╗"
    corner_bl = "╚"
    corner_br = "╝"
    side = "║"
    divider_left = "╠"
    divider_right = "╣"
    
    # Title line
    title_text = f"{emoji} {title}" if emoji else title
    title_line = f"{side} {title_text.center(width - 4)} {side}"
    
    # Print box
    print(f"{corner_tl}{border_char * (width - 2)}{corner_tr}")
    print(title_line)
    print(f"{divider_left}{border_char * (width - 2)}{divider_right}")
    
    for line in content:
        # Handle long lines by wrapping
        if len(line) <= width - 4:
            print(f"{side} {line.ljust(width - 4)} {side}")
        else:
            # Wrap long lines
            words = line.split()
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) <= width - 4:
                    current_line = test_line
                else:
                    if current_line:
                        print(f"{side} {current_line.ljust(width - 4)} {side}")
                    current_line = word
            if current_line:
                print(f"{side} {current_line.ljust(width - 4)} {side}")
    
    print(f"{corner_bl}{border_char * (width - 2)}{corner_br}")
    sys.stdout.flush()


def log_retrieval_results(query: str, elements: List[Dict[str, Any]], 
                          max_preview_length: int = 80):
    """
    Log retrieval results in a formatted box.
    
    Args:
        query: The search query
        elements: List of retrieved elements with scores
        max_preview_length: Maximum length for code preview
    """
    if not _VERBOSE_LOGGING or not elements:
        return
    
    content = []
    content.append(f'Query: "{query[:100]}{"..." if len(query) > 100 else ""}"')
    content.append(f"Retrieved {len(elements)} elements:")
    content.append("")
    
    for i, elem_data in enumerate(elements[:10], 1):  # Limit to 10 elements
        elem = elem_data.get("element", {})
        score = elem_data.get("total_score", 0)
        
        elem_type = elem.get("type", "unknown").upper()
        name = elem.get("name", "unknown")
        repo = elem.get("repo_name", "")
        rel_path = elem.get("relative_path", elem.get("file_path", ""))
        start_line = elem.get("start_line", 0)
        
        # Format element header
        location = f"{rel_path}:{start_line}" if start_line else rel_path
        if repo:
            location = f"{repo}/{location}"
        
        content.append(f"  {i}. [{elem_type}] {name} - {location}")
        content.append(f"     Score: {score:.2f}")
        
        # Add code preview
        code = elem.get("code", "")
        if code:
            preview = code.split("\n")[0][:max_preview_length]
            if len(code.split("\n")[0]) > max_preview_length:
                preview += "..."
            content.append(f"     Preview: {preview}")
        content.append("")
    
    if len(elements) > 10:
        content.append(f"  ... and {len(elements) - 10} more elements")
    
    _print_box("RETRIEVAL RESULTS", content, emoji="🔍")


def log_llm_request(model: str, messages: List[Dict[str, str]], 
                    base_url: Optional[str] = None):
    """
    Log LLM request in a formatted box.
    
    Args:
        model: Model name
        messages: List of messages (role, content)
        base_url: API endpoint URL
    """
    if not _VERBOSE_LOGGING:
        return
    
    content = []
    content.append(f"Model: {model}")
    if base_url:
        content.append(f"Endpoint: {base_url}")
    content.append("Messages:")
    content.append("─" * 50)
    
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        msg_content = msg.get("content", "")
        
        content.append(f"[{role}]")
        
        # Show full content without truncation
        lines = msg_content.split("\n")
        for line in lines:
            content.append(line)
        content.append("─" * 50)
    
    _print_box("LLM REQUEST", content, emoji="📤")


def log_llm_response(response: str, prompt_tokens: Optional[int] = None,
                     completion_tokens: Optional[int] = None,
                     total_tokens: Optional[int] = None):
    """
    Log LLM response in a formatted box.
    
    Args:
        response: Response content from LLM
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens used
    """
    if not _VERBOSE_LOGGING:
        return
    
    content = []
    
    # Token usage
    if total_tokens or prompt_tokens or completion_tokens:
        token_info = []
        if total_tokens:
            token_info.append(f"total: {total_tokens}")
        if prompt_tokens:
            token_info.append(f"prompt: {prompt_tokens}")
        if completion_tokens:
            token_info.append(f"completion: {completion_tokens}")
        content.append(f"Tokens Used: {', '.join(token_info)}")
        content.append("─" * 50)
    
    # Response content - show full content without truncation
    lines = response.split("\n")
    for line in lines:
        content.append(line)
    
    _print_box("LLM RESPONSE", content, emoji="📥")


def log_agent_round(round_num: int, confidence: int, tool_calls: List[Dict[str, Any]] = None,
                    elements_count: int = 0, total_lines: int = 0):
    """
    Log agent round information.
    
    Args:
        round_num: Round number
        confidence: Confidence score (0-100)
        tool_calls: List of tool calls made
        elements_count: Number of elements retrieved
        total_lines: Total lines of code
    """
    if not _VERBOSE_LOGGING:
        return
    
    content = []
    content.append(f"Round: {round_num}")
    content.append(f"Confidence: {confidence}%")
    content.append(f"Elements: {elements_count}")
    content.append(f"Total Lines: {total_lines}")
    
    if tool_calls:
        content.append("")
        content.append("Tool Calls:")
        for i, tc in enumerate(tool_calls[:5], 1):
            tool_name = tc.get("tool", "unknown")
            params = tc.get("parameters", {})
            content.append(f"  {i}. {tool_name}: {params}")
        if len(tool_calls) > 5:
            content.append(f"  ... and {len(tool_calls) - 5} more")
    
    _print_box(f"AGENT ROUND {round_num}", content, emoji="🔄")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get("file", "./logs/fastcode.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    handlers = []
    if log_config.get("console", True):
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    return logging.getLogger("fastcode")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""


def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(512)
        return True
    except (UnicodeDecodeError, IOError):
        return False


def get_file_extension(file_path: str) -> str:
    """Get file extension"""
    return Path(file_path).suffix


def is_supported_file(file_path: str, supported_extensions: List[str]) -> bool:
    """Check if file extension is supported"""
    ext = get_file_extension(file_path)
    return ext in supported_extensions


def should_ignore_path(path: str, ignore_patterns: List[str]) -> bool:
    """Check if path should be ignored based on patterns"""
    from pathspec import PathSpec
    from pathspec.patterns import GitWildMatchPattern
    
    spec = PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)
    return spec.match_file(path)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Some retrieved snippets may contain literal special-token strings like
    # "<|endoftext|>", which raise in tiktoken.encode by default. Allow them so
    # counting doesn't fail on the first query in non-English cases.
    return len(encoding.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def normalize_path(path: str) -> str:
    """Normalize file path"""
    return os.path.normpath(path).replace("\\", "/")


def get_language_from_extension(ext: str) -> str:
    """Get programming language from file extension"""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objective-c",
        ".sh": "bash",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
    }
    return language_map.get(ext.lower(), "unknown")


def extract_code_snippet(content: str, start_line: int, end_line: int, 
                         context_lines: int = 3) -> Dict[str, Any]:
    """Extract code snippet with context"""
    lines = content.split("\n")
    total_lines = len(lines)
    
    # Calculate actual range with context
    actual_start = max(0, start_line - context_lines)
    actual_end = min(total_lines, end_line + context_lines)
    
    snippet_lines = lines[actual_start:actual_end]
    
    return {
        "code": "\n".join(snippet_lines),
        "start_line": actual_start + 1,  # 1-indexed
        "end_line": actual_end,
        "highlighted_start": start_line + 1,
        "highlighted_end": end_line,
    }


def format_code_block(code: str, language: str = "", file_path: str = "", 
                      start_line: Optional[int] = None) -> str:
    """Format code block for display"""
    header = f"```{language}"
    if file_path:
        header += f" - {file_path}"
    if start_line:
        header += f" (Line {start_line})"
    
    return f"{header}\n{code}\n```"


def calculate_code_complexity(code: str) -> int:
    """Calculate simple cyclomatic complexity"""
    # Simple heuristic: count control flow keywords
    keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 
                'catch', 'case', 'switch', '&&', '||', '?']
    
    complexity = 1
    for keyword in keywords:
        complexity += code.count(keyword)
    
    return complexity


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text with sliding window"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "start_word": i,
            "end_word": i + len(chunk_words),
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_get(d: Dict, *keys, default=None):
    """Safely get nested dictionary value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
            if d is None:
                return default
        else:
            return default
    return d


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_repo_name_from_url(url: str) -> str:
    """Extract repository name from URL"""
    # Handle GitHub URLs
    if url.endswith(".git"):
        url = url[:-4]
    
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts else "unknown_repo"


def clean_docstring(docstring: str) -> str:
    """Clean and format docstring"""
    if not docstring:
        return ""
    
    lines = docstring.split("\n")
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    # Find minimum indentation
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # Remove common indentation
    if min_indent < float('inf'):
        lines = [line[min_indent:] if len(line) > min_indent else line 
                 for line in lines]
    
    return "\n".join(lines).strip()

