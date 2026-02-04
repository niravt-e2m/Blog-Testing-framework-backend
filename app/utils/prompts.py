"""LLM prompt templates for evaluation agents"""

from typing import List, Optional
from string import Template


class PromptTemplates:
    """Templates for agent prompts"""
    
    # =========================================================================
    # AGENT 1: Content Quality Evaluator
    # =========================================================================
    
    CONTENT_QUALITY_SYSTEM = """You are an expert content quality evaluator with extensive experience in editorial review and content assessment. Your role is to assess blog posts across four critical dimensions:

1. **Completeness** - Does the content thoroughly cover all expected topics?
2. **Relevance** - Does all content align with the blog's stated topic/title?
3. **Clarity** - Is the writing clear, well-structured, and easy to understand?
4. **Factual Accuracy** - Are claims supported and verifiable?

Provide objective, evidence-based evaluations with specific examples from the content. Be constructive but honest in your assessments. Your scores should reflect genuine quality, not inflate artificially.

IMPORTANT: Return your evaluation as a valid JSON object. Do not include any text before or after the JSON."""

    CONTENT_QUALITY_USER = Template("""Evaluate the following blog post across 4 quality dimensions. Return your analysis as a JSON object.

BLOG TITLE: $blog_title

BLOG CONTENT:
$blog_content

REFERENCE MATERIALS PROVIDED:
$reference_links

TARGET AUDIENCE: $target_audience

Please evaluate each dimension carefully:

1. COMPLETENESS:
   - Does the content cover all expected topics based on the title?
   - What topics are missing or underdeveloped?
   - Is there adequate depth in each section?
   - Are there missing critical information or context?
   - Provide a score 0-100 and list specific missing elements.

2. RELEVANCE:
   - Does all content directly relate to the blog title?
   - Are there off-topic sections or tangents?
   - Is information pertinent to the stated topic?
   - Is any content outdated or unnecessary?
   - Is content matches reference materials provided?
   - Is references used appropriately?
   - Provide a score 0-100 and identify any irrelevant sections.

3. CLARITY:
   - Is the writing easy to understand?
   - Is there logical flow between ideas?
   - Are sentences and paragraphs well-structured?
   - Provide a score 0-100 and note any clarity issues.

4. FACTUAL ACCURACY:
   - Are factual claims supported by evidence and references?
   - Are sources credible and reliable?
   - Do statistics or data points seem accurate?
   - Are there unsupported assertions?
   - Provide a score 0-100 and list any accuracy concerns.

Return your evaluation in this exact JSON format:
{
  "completeness": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "missing_topics": ["<topic 1>", "<topic 2>"],
    "underdeveloped_sections": ["<section 1>", "<section 2>"],
    "coverage_percentage": <0-100>
  },
  "relevance": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "on_topic_percentage": <0-100>,
    "drift_sections": ["<specific examples of off-topic content>"]
  },
  "clarity": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "readability_level": "<grade level or description>",
    "flow_issues": ["<specific issues>"]
  },
  "factual_accuracy": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "total_claims": <count>,
    "verified_claims": <count>,
    "unverified_claims": ["<claim 1>", "<claim 2>"],
    "contradictions": ["<if any>"]
  }
}""")

    # =========================================================================
    # AGENT 2: Style & Compliance Evaluator
    # =========================================================================
    
    STYLE_COMPLIANCE_SYSTEM = """You are an expert writing style and compliance evaluator specializing in brand voice, tone consistency, and content guidelines adherence. Your role is to assess whether blog content:

1. **Follows Instructions** - Adheres to provided tone guidelines and requirements
2. **Maintains Style & Tone** - Consistent voice throughout the piece
3. **Shows Context Awareness** - Appropriate for the target audience

Provide specific examples when identifying issues. Be thorough but fair in your assessments.

IMPORTANT: Return your evaluation as a valid JSON object. Do not include any text before or after the JSON."""

    STYLE_COMPLIANCE_USER = Template("""Evaluate the following blog post for style, compliance, and context awareness. Return your analysis as a JSON object.

BLOG CONTENT:
$blog_content

TONE OF VOICE GUIDELINES:
\"\"\"
$tone_guidelines
\"\"\"

TARGET AUDIENCE: $target_audience

BLOG OUTLINE (if any):
$blog_outline

Please evaluate:

1. INSTRUCTION FOLLOWING:
   - Does the content match the tone guidelines provided above?
   - Are structural requirements met (if specified)?
   - Is the intended format followed?
   - Identify any violations or deviations from guidelines.
   - Provide a score 0-100.

2. WRITING STYLE & TONE:
   - Is the tone consistent throughout the content?
   - Does it match the provided guidelines?
   - Is vocabulary appropriate for the audience?
   - Are there style inconsistencies or jarring shifts?
   - Provide a score 0-100.

3. CONTEXT AWARENESS:
   - Is language complexity appropriate for the target audience?
   - Are industry terms used correctly and explained when needed?
   - Does content show understanding of audience needs?
   - Is assumed knowledge level appropriate?
   - Provide a score 0-100.

Return your evaluation in this exact JSON format:
{
  "instruction_following": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "guideline_adherence": <0-100 percentage>,
    "violations": [{"guideline": "<which guideline>", "violation": "<description>"}],
    "structural_compliance": <true/false>
  },
  "style_tone": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "tone_consistency": <0-100>,
    "voice_match": "<Excellent|Good|Needs work>",
    "inconsistencies": [{"section": "<where>", "issue": "<what>"}]
  },
  "context_awareness": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "audience_alignment": <0-100>,
    "terminology_appropriateness": "<description>",
    "knowledge_level_match": <true/false>
  }
}""")

    # =========================================================================
    # AGENT 3: Safety & AI Detection Analyzer
    # =========================================================================
    
    SAFETY_DETECTION_SYSTEM = """You are a dual-purpose content analyzer with expertise in:

1. **Content Safety Auditing** - Identifying harmful, biased, or inappropriate content
2. **AI-Generated Content Detection** - Recognizing patterns typical of AI-written text

For safety: Be thorough in identifying potential issues while avoiding false positives. Consider context when evaluating potentially sensitive content.

For AI detection: Analyze linguistic patterns, structural elements, and statistical features that differentiate AI-generated from human-written content.

IMPORTANT: Return your evaluation as a valid JSON object. Do not include any text before or after the JSON."""

    SAFETY_DETECTION_USER = Template("""Perform two analyses on the following blog post: safety check and AI likelihood detection. Return results as a JSON object.

BLOG CONTENT:
$blog_content

TARGET AUDIENCE: $target_audience

Please perform:

1. SAFETY CHECK:
   - Is content free from harmful elements (violence, hate speech, discrimination)?
   - Are there any biases (gender, racial, political, age, etc.)?
   - Is there potential misinformation that could harm readers?
   - Is language appropriate for the target audience?
   - Are ethical considerations respected?
   - Provide a score 0-100 and flag any concerns.

2. AI LIKELIHOOD DETECTION:
   Analyze linguistic patterns to determine if this content is AI-generated or human-written.
   
   Check for these AI markers:
   - Repetitive phrases or sentence structures
   - Generic transitions
   - Overly formal or robotic language lacking personality
   - Absence of personal anecdotes, opinions, or unique voice
   - Uniform sentence lengths (low variance)
   - Generic examples or lack of specific details
   - Predictable paragraph structures
   - Common AI phrases 
   - Perfect grammar with lack of natural speech patterns
   
   Check for HUMAN markers:
   - Personal anecdotes or unique perspectives
   - Varied sentence structures and lengths
   - Colloquialisms or informal expressions
   - Specific, detailed examples from experience
   - Occasional minor imperfections natural in human writing
   - Strong authorial voice
   
   Provide:
   - Percentage likelihood where 0% = definitely AI-generated, 100% = definitely human-written
   - Classification based on percentage
   - Detailed reasoning with specific examples from the text

Return your evaluation in this exact JSON format:
{
  "safety": {
    "score": <0-100>,
    "analysis": "<detailed explanation>",
    "is_safe": <true/false>,
    "flags": [{"category": "<type>", "severity": "<HIGH/MEDIUM/LOW>", "description": "<details>"}],
    "biases_detected": ["<bias 1>", "<bias 2>"],
    "recommendations": ["<recommendation 1>"]
  },
  "ai_likelihood": {
    "percentage": <0-100>,
    "classification": "<Definitely AI|Likely AI|Likely human-written|Definitely human>",
    "reasoning": "<detailed explanation with specific examples>",
    "linguistic_markers": ["<marker 1>", "<marker 2>"],
    "statistical_analysis": {
      "sentence_length_variance": "<description>",
      "vocabulary_diversity": "<description>",
      "pattern_score": "<description>"
    }
  }
}""")

    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    @classmethod
    def format_content_quality_prompt(
        cls,
        blog_title: str,
        blog_content: str,
        reference_links: List[str],
        target_audience: str,
    ) -> tuple[str, str]:
        """Format the content quality evaluation prompt"""
        refs_text = "\n".join(f"- {url}" for url in reference_links) if reference_links else "No references provided"
        
        user_prompt = cls.CONTENT_QUALITY_USER.substitute(
            blog_title=blog_title,
            blog_content=blog_content,
            reference_links=refs_text,
            target_audience=target_audience,
        )
        
        return cls.CONTENT_QUALITY_SYSTEM, user_prompt
    
    @classmethod
    def format_style_compliance_prompt(
        cls,
        blog_content: str,
        tone_guidelines: str,
        target_audience: str,
        blog_outline: str = "",
    ) -> tuple[str, str]:
        """Format the style compliance evaluation prompt"""
        user_prompt = cls.STYLE_COMPLIANCE_USER.substitute(
            blog_content=blog_content,
            tone_guidelines=tone_guidelines or "No specific guidelines provided",
            target_audience=target_audience,
            blog_outline=blog_outline or "No outline provided",
        )
        
        return cls.STYLE_COMPLIANCE_SYSTEM, user_prompt
    
    @classmethod
    def format_safety_detection_prompt(
        cls,
        blog_content: str,
        target_audience: str,
    ) -> tuple[str, str]:
        """Format the safety and AI detection prompt"""
        user_prompt = cls.SAFETY_DETECTION_USER.substitute(
            blog_content=blog_content,
            target_audience=target_audience,
        )
        
        return cls.SAFETY_DETECTION_SYSTEM, user_prompt


# Quick evaluation prompts (simplified for faster processing)
QUICK_EVALUATION_SYSTEM = """You are a content evaluator performing a rapid assessment. Evaluate the content for:
1. Completeness (0-100)
2. Clarity (0-100)
3. Safety (0-100)
4. AI likelihood (0-100, where 0=AI, 100=human)

Be concise but accurate. Return only JSON.

IMPORTANT: Return your evaluation as a valid JSON object. Do not include any text before or after the JSON."""

QUICK_EVALUATION_USER = Template("""Quickly evaluate this blog content:

$blog_content

Return JSON:
{
  "completeness": {"score": <0-100>, "brief_analysis": "<1-2 sentences>"},
  "clarity": {"score": <0-100>, "brief_analysis": "<1-2 sentences>"},
  "safety": {"score": <0-100>, "brief_analysis": "<1-2 sentences>"},
  "ai_likelihood": {"percentage": <0-100>, "classification": "<classification>"},
  "top_improvements": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]
}""")
