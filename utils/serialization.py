def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif hasattr(obj, "page_content"):
        return {"page_content": obj.page_content, "metadata": obj.metadata}
    elif hasattr(obj, "__dict__"):
        return str(obj)
    else:
        return obj
