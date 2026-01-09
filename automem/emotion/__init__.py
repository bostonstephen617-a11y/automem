"""
AutoMem Emotion Analysis Module

Provides emotion-aware capabilities for the memory system based on
the hierarchical emotion organization methodology from:
"Emergence of Hierarchical Emotion Organization in Large Language Models"

This module is designed for non-disruptive integration:
- Optional: All existing functionality works without emotion analysis
- Backward compatible: Existing memory schemas remain valid
- Graceful degradation: System continues working if emotion analysis fails
"""

from .analysis import EmotionAnalyzer, EmotionHierarchy, EmotionAnalysisResult
from .bias_detection import EmotionBiasDetector
from .config import (
    EMOTION_ANALYSIS_ENABLED,
    EMOTION_LEXICON_135,
    EMOTION_HIERARCHY_THRESHOLD
)

__all__ = [
    "EmotionAnalyzer",
    "EmotionHierarchy", 
    "EmotionAnalysisResult",
    "EmotionBiasDetector",
    "EMOTION_ANALYSIS_ENABLED",
    "EMOTION_LEXICON_135",
    "EMOTION_HIERARCHY_THRESHOLD"
]