"""
AI-Powered Validation Agent
Uses AI to validate extracted information and detect hallucinations
"""

from typing import Dict, Any, List
from .prompts import PROMPTS
from .config import settings
from .token_utils import TokenTracker
from langchain_community.llms import Ollama


class AIValidator:
    """AI agent that validates extracted information and detects hallucinations"""
    
    def __init__(self):
        self.validation_prompt = PROMPTS["validation"]
    
    def validate_extraction(self, extracted_data: Dict[str, Any], original_content: str, session_id: str = "default") -> Dict[str, Any]:
        """Validate extracted information using AI"""
        try:
            # Build validation prompt
            validation_prompt = self._build_validation_prompt(extracted_data, original_content)
            
            extracted_summary = self._summarize_extracted_data(extracted_data)
            content_truncated = original_content[:3000]
            
            # Get AI validation
            _llm = Ollama(model=settings.ollama_general_model, base_url=settings.ollama_base_url, temperature=0.2)
            
            with TokenTracker(
                operation="validate_extraction",
                session_id=session_id,
                prompt=validation_prompt,
                content=content_truncated,
                reference_content=extracted_summary
            ) as tracker:
                tracker.mark_generation_start()
                response = _llm.invoke(validation_prompt)
                result = str(response)
                tracker.mark_generation_done(result)
            
            # Parse validation response
            validation_result = self._parse_validation_response(result)
            
            return validation_result
            
        except Exception as e:
            return self._fallback_validation(str(e))
    
    def _build_validation_prompt(self, extracted_data: Dict[str, Any], original_content: str) -> str:
        """Build validation prompt with extracted data and original content"""
        
        # Prepare extracted data for validation
        extracted_summary = self._summarize_extracted_data(extracted_data)
        
        prompt = f"""
        {self.validation_prompt}
        
        ORIGINAL CONTENT:
        {original_content[:3000]}  # Limit for token efficiency
        
        EXTRACTED INFORMATION:
        {extracted_summary}
        
        Please validate the extracted information against the original content.
        Check for accuracy, completeness, and potential hallucinations.
        """
        
        return prompt
    
    def _summarize_extracted_data(self, extracted_data: Dict[str, Any]) -> str:
        """Summarize extracted data for validation"""
        summary_parts = []
        
        # Add each section if it exists
        for section, data in extracted_data.items():
            if isinstance(data, dict) and data:
                summary_parts.append(f"{section.upper()}:")
                for key, value in data.items():
                    if value is not None and value != "" and value != []:
                        summary_parts.append(f"  {key}: {value}")
                summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse AI validation response"""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_validation("Could not parse validation response")
        except:
            return self._fallback_validation("JSON parsing failed")
    
    def _fallback_validation(self, error: str) -> Dict[str, Any]:
        """Fallback validation when AI analysis fails"""
        return {
            "validation_results": {
                "is_accurate": False,
                "confidence_score": 0.0,
                "completeness_score": 0.0,
                "hallucination_risk": "high"
            },
            "accuracy_check": {
                "verified_facts": [],
                "potential_errors": [f"Validation failed: {error}"],
                "missing_verification": ["all_extracted_data"]
            },
            "hallucination_detection": {
                "suspicious_claims": ["Unable to verify due to validation failure"],
                "confidence_issues": ["Validation system error"],
                "source_verification": ["Manual verification required"]
            },
            "completeness_assessment": {
                "missing_important_info": ["Validation system error"],
                "incomplete_sections": ["all_sections"],
                "suggestions": ["Manual validation recommended"]
            },
            "overall_assessment": {
                "quality_rating": "poor",
                "reliability": "low",
                "recommendations": ["Fix validation system", "Manual review required"]
            }
        }
