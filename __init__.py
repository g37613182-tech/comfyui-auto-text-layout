from .auto_text_layout import AutoTextLayout

NODE_CLASS_MAPPINGS = {
    "AutoTextLayout": AutoTextLayout,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoTextLayout": "Auto Text Layout ✏️",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
