"""
Stub template manager for initial implementation.
"""

class TemplateManager:
    def __init__(self):
        pass
    
    def has_template(self, template_name):
        return False
    
    def get_template(self, template_name):
        raise ValueError(f"Template {template_name} not found")
    
    def list_templates(self):
        return []
    
    def get_template_info(self, template_name):
        return {"description": "Template not found"}