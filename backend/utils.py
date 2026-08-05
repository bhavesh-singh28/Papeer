def safe_get_text(content) -> str:
    """Safely extract plain text from LangChain message content.
    
    Handles strings, lists of content blocks (for multimodal/complex layout support),
    and dictionaries representing text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = []
        for part in content:
            if isinstance(part, str):
                text.append(part)
            elif isinstance(part, dict) and "text" in part:
                text.append(part["text"])
            elif hasattr(part, "get") and part.get("text"):
                text.append(part.get("text"))
        return "".join(text)
    return str(content) if content is not None else ""
