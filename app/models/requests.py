"""Pydantic request models for API endpoints"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ToneOfVoiceInput(BaseModel):
    """Tone of voice input specification"""
    type: Literal["text", "file"] = "text"
    content: str = Field(..., description="The tone of voice guidelines or file content")
    file_name: Optional[str] = Field(None, alias="fileName")
    
    class Config:
        populate_by_name = True


class Reference(BaseModel):
    """Reference link or text for fact-checking"""
    id: str = Field(..., description="Unique identifier for the reference")
    type: Literal["link", "text"] = Field(..., description="Type of reference")
    content: str = Field(..., description="URL or text content")


class EvaluationOptions(BaseModel):
    """Options for customizing evaluation behavior"""
    async_mode: bool = Field(False, description="Run evaluation asynchronously")
    include_ai_detection: bool = Field(True, description="Include AI likelihood detection")
    depth_level: Literal["fast", "standard", "deep"] = Field(
        "standard", description="Evaluation depth level"
    )


class EvaluationRequest(BaseModel):
    """Main request model for blog evaluation"""
    
    # Core content fields (matching frontend EvaluationInput)
    blog_title: str = Field(
        ...,
        alias="blogTitle",
        description="Blog title"
    )
    blog_text: str = Field(
        ..., 
        alias="blogText",
        min_length=100,
        description="The blog content to evaluate"
    )
    blog_outline: str = Field(
        "",
        alias="blogOutline",
        description="The blog outline or structure"
    )
    tone_of_voice: ToneOfVoiceInput = Field(
        ...,
        alias="toneOfVoice",
        description="Tone of voice guidelines"
    )
    references: List[Reference] = Field(
        default_factory=list,
        description="Reference links or text for fact-checking"
    )
    target_audience: str = Field(
        ...,
        alias="targetAudience",
        description="Target audience description"
    )
    
    # Legacy/alternative field names for compatibility
    blog_content: Optional[str] = Field(
        None,
        alias="blogContent",
        description="Alternative field name for blog text"
    )
    tone_guidelines: Optional[str] = Field(
        None,
        alias="toneGuidelines",
        description="Alternative field name for tone of voice content"
    )
    reference_links: Optional[List[str]] = Field(
        None,
        alias="referenceLinks",
        description="Legacy reference links format"
    )
    
    # Evaluation options
    evaluation_options: Optional[EvaluationOptions] = Field(
        None,
        alias="evaluationOptions",
        description="Options for customizing evaluation"
    )
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "blogTitle": "10 Python Tips for Data Scientists",
                "blogText": "Python is essential for data science. In this comprehensive guide, we'll explore 10 powerful tips that will transform your data science workflow...",
                "blogOutline": "1) Intro 2) Tip 1 ... 10) Conclusion",
                "toneOfVoice": {
                    "type": "text",
                    "content": "Professional but approachable. Use clear examples and avoid jargon."
                },
                "references": [
                    {
                        "id": "ref1",
                        "type": "link",
                        "content": "https://docs.python.org/3/"
                    }
                ],
                "targetAudience": "Intermediate data scientists looking to improve their Python skills",
                "evaluationOptions": {
                    "async_mode": False,
                    "include_ai_detection": True,
                    "depth_level": "standard"
                }
            }
        }
    
    @model_validator(mode='after')
    def validate_content_fields(self):
        """Ensure we have the required content from either field name"""
        # Use blog_content if blog_text is not provided
        if not self.blog_text and self.blog_content:
            self.blog_text = self.blog_content
        
        # Use tone_guidelines if tone_of_voice content is empty
        if self.tone_guidelines and not self.tone_of_voice.content:
            self.tone_of_voice = ToneOfVoiceInput(
                type="text",
                content=self.tone_guidelines
            )
        
        # Convert legacy reference_links to references
        if self.reference_links and not self.references:
            self.references = [
                Reference(id=f"ref_{i}", type="link", content=url)
                for i, url in enumerate(self.reference_links)
            ]
        
        return self
    
    @property
    def content(self) -> str:
        """Get the blog content"""
        return self.blog_text or self.blog_content or ""
    
    @property
    def title(self) -> str:
        """Get the blog title, extracting from content if needed"""
        if self.blog_title:
            return self.blog_title
        # Try to extract first line as title
        lines = self.content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove markdown heading markers
            if first_line.startswith('#'):
                return first_line.lstrip('#').strip()
            return first_line[:100]  # Limit title length
        return "Untitled"

    @property
    def outline(self) -> str:
        """Get the blog outline"""
        return self.blog_outline or ""
    
    @property
    def tone_content(self) -> str:
        """Get the tone of voice content"""
        if self.tone_of_voice and self.tone_of_voice.content:
            return self.tone_of_voice.content
        return self.tone_guidelines or ""
    
    @property
    def reference_urls(self) -> List[str]:
        """Get all reference URLs"""
        urls = []
        for ref in self.references:
            if ref.type == "link":
                urls.append(ref.content)
        if self.reference_links:
            urls.extend(self.reference_links)
        return list(set(urls))  # Deduplicate
    
    @property
    def reference_texts(self) -> List[str]:
        """Get all reference text content"""
        return [ref.content for ref in self.references if ref.type == "text"]


class QuickEvaluationRequest(BaseModel):
    """Simplified request for quick evaluation (3 core metrics only)"""
    
    blog_text: str = Field(
        ...,
        alias="blogText",
        min_length=100,
        description="The blog content to evaluate"
    )
    tone_of_voice: Optional[str] = Field(
        None,
        alias="toneOfVoice",
        description="Optional tone of voice guidelines"
    )
    target_audience: Optional[str] = Field(
        None,
        alias="targetAudience",
        description="Optional target audience description"
    )
    
    class Config:
        populate_by_name = True


class ReferenceValidationRequest(BaseModel):
    """Request for standalone reference validation"""
    
    urls: List[str] = Field(
        ...,
        min_length=1,
        description="List of URLs to validate"
    )
