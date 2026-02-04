"""Text analysis utilities for readability and statistical metrics"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class TextStatistics:
    """Container for text statistics"""
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    syllable_count: int
    complex_word_count: int  # Words with 3+ syllables
    unique_word_count: int
    avg_word_length: float
    avg_sentence_length: float
    avg_paragraph_length: float


@dataclass
class ReadabilityScores:
    """Container for readability metrics"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    avg_grade_level: float
    readability_level: str  # "Easy", "Standard", "Difficult"


class TextAnalyzer:
    """Analyzer for text statistics and readability metrics"""
    
    # Common word patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]+')
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n|\r\n\s*\r\n')
    
    # Vowel pattern for syllable counting
    VOWELS = 'aeiouy'
    VOWEL_GROUPS = re.compile(r'[aeiouy]+', re.IGNORECASE)
    
    # Silent 'e' and common exceptions
    SILENT_E = re.compile(r'[^aeiouy]e$', re.IGNORECASE)
    ENDINGS_WITH_SYLLABLE = re.compile(r'(le|les|ed|es)$', re.IGNORECASE)
    
    def __init__(self, text: str):
        """Initialize analyzer with text content"""
        self.original_text = text
        self.text = self._normalize_text(text)
        self._words: Optional[List[str]] = None
        self._sentences: Optional[List[str]] = None
        self._paragraphs: Optional[List[str]] = None
        self._stats: Optional[TextStatistics] = None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove markdown headers and formatting
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @property
    def words(self) -> List[str]:
        """Get list of words"""
        if self._words is None:
            self._words = self.WORD_PATTERN.findall(self.text.lower())
        return self._words
    
    @property
    def sentences(self) -> List[str]:
        """Get list of sentences"""
        if self._sentences is None:
            # Split on sentence endings but keep the content
            raw_sentences = self.SENTENCE_ENDINGS.split(self.text)
            self._sentences = [s.strip() for s in raw_sentences if s.strip()]
        return self._sentences
    
    @property
    def paragraphs(self) -> List[str]:
        """Get list of paragraphs"""
        if self._paragraphs is None:
            raw_paragraphs = self.PARAGRAPH_PATTERN.split(self.original_text)
            self._paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
        return self._paragraphs
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower().strip()
        if len(word) <= 2:
            return 1
        
        # Count vowel groups
        vowel_groups = self.VOWEL_GROUPS.findall(word)
        count = len(vowel_groups)
        
        # Subtract silent 'e' at end
        if self.SILENT_E.search(word) and count > 1:
            count -= 1
        
        # Handle special endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in self.VOWELS:
            count += 1
        
        # Minimum of 1 syllable
        return max(1, count)
    
    def is_complex_word(self, word: str) -> bool:
        """Check if word is complex (3+ syllables)"""
        return self.count_syllables(word) >= 3
    
    def get_statistics(self) -> TextStatistics:
        """Calculate comprehensive text statistics"""
        if self._stats is not None:
            return self._stats
        
        words = self.words
        sentences = self.sentences
        paragraphs = self.paragraphs
        
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        paragraph_count = max(len(paragraphs), 1)
        
        # Calculate syllable counts
        syllable_count = sum(self.count_syllables(w) for w in words)
        complex_words = [w for w in words if self.is_complex_word(w)]
        
        # Calculate unique words
        unique_words = set(words)
        
        # Calculate averages
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        avg_sentence_length = word_count / sentence_count
        avg_paragraph_length = word_count / paragraph_count
        
        self._stats = TextStatistics(
            character_count=len(self.text),
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            syllable_count=syllable_count,
            complex_word_count=len(complex_words),
            unique_word_count=len(unique_words),
            avg_word_length=round(avg_word_length, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            avg_paragraph_length=round(avg_paragraph_length, 2),
        )
        return self._stats
    
    def calculate_flesch_reading_ease(self) -> float:
        """
        Calculate Flesch Reading Ease score.
        Formula: 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)
        Score range: 0-100 (higher = easier to read)
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        words_per_sentence = stats.word_count / stats.sentence_count
        syllables_per_word = stats.syllable_count / stats.word_count
        
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        return round(max(0, min(100, score)), 2)
    
    def calculate_flesch_kincaid_grade(self) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        Formula: 0.39 × (words/sentences) + 11.8 × (syllables/words) - 15.59
        Returns grade level (e.g., 8.5 = 8th-9th grade)
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        words_per_sentence = stats.word_count / stats.sentence_count
        syllables_per_word = stats.syllable_count / stats.word_count
        
        grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
        return round(max(0, grade), 2)
    
    def calculate_gunning_fog_index(self) -> float:
        """
        Calculate Gunning Fog Index.
        Formula: 0.4 × [(words/sentences) + 100 × (complex_words/words)]
        Returns grade level
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        words_per_sentence = stats.word_count / stats.sentence_count
        complex_word_ratio = stats.complex_word_count / stats.word_count
        
        index = 0.4 * (words_per_sentence + (100 * complex_word_ratio))
        return round(max(0, index), 2)
    
    def calculate_smog_index(self) -> float:
        """
        Calculate SMOG (Simple Measure of Gobbledygook) Index.
        Formula: 1.0430 × √(complex_words × 30/sentences) + 3.1291
        Returns grade level
        """
        stats = self.get_statistics()
        if stats.sentence_count < 3:
            # SMOG requires at least 30 sentences ideally, but we'll adapt
            return self.calculate_flesch_kincaid_grade()
        
        polysyllable_count = stats.complex_word_count
        
        # Normalize to 30 sentences
        normalized_polysyllables = polysyllable_count * (30 / stats.sentence_count)
        
        index = 1.0430 * math.sqrt(normalized_polysyllables) + 3.1291
        return round(max(0, index), 2)
    
    def calculate_coleman_liau_index(self) -> float:
        """
        Calculate Coleman-Liau Index.
        Formula: 0.0588 × L - 0.296 × S - 15.8
        Where L = avg letters per 100 words, S = avg sentences per 100 words
        Returns grade level
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        # Letters per 100 words
        L = (stats.character_count / stats.word_count) * 100
        # Sentences per 100 words
        S = (stats.sentence_count / stats.word_count) * 100
        
        index = (0.0588 * L) - (0.296 * S) - 15.8
        return round(max(0, index), 2)
    
    def calculate_automated_readability_index(self) -> float:
        """
        Calculate Automated Readability Index (ARI).
        Formula: 4.71 × (characters/words) + 0.5 × (words/sentences) - 21.43
        Returns grade level
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        chars_per_word = stats.character_count / stats.word_count
        words_per_sentence = stats.word_count / stats.sentence_count
        
        index = (4.71 * chars_per_word) + (0.5 * words_per_sentence) - 21.43
        return round(max(0, index), 2)
    
    def calculate_vocabulary_diversity(self) -> float:
        """
        Calculate Type-Token Ratio (vocabulary diversity).
        Returns ratio (0-1, higher = more diverse vocabulary)
        """
        stats = self.get_statistics()
        if stats.word_count == 0:
            return 0.0
        
        return round(stats.unique_word_count / stats.word_count, 4)
    
    def analyze_sentence_lengths(self) -> Dict[str, float]:
        """
        Analyze sentence length distribution.
        Returns statistics about sentence lengths.
        """
        if not self.sentences:
            return {
                "mean": 0,
                "median": 0,
                "std_deviation": 0,
                "min": 0,
                "max": 0,
                "variance": 0,
            }
        
        # Count words per sentence
        lengths = [len(self.WORD_PATTERN.findall(s)) for s in self.sentences]
        
        n = len(lengths)
        mean = sum(lengths) / n
        
        # Median
        sorted_lengths = sorted(lengths)
        if n % 2 == 0:
            median = (sorted_lengths[n//2 - 1] + sorted_lengths[n//2]) / 2
        else:
            median = sorted_lengths[n//2]
        
        # Standard deviation and variance
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std_dev = math.sqrt(variance)
        
        return {
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std_deviation": round(std_dev, 2),
            "min": min(lengths),
            "max": max(lengths),
            "variance": round(variance, 2),
        }
    
    def get_readability_scores(self) -> ReadabilityScores:
        """Get all readability metrics"""
        flesch_ease = self.calculate_flesch_reading_ease()
        flesch_kincaid = self.calculate_flesch_kincaid_grade()
        gunning_fog = self.calculate_gunning_fog_index()
        smog = self.calculate_smog_index()
        coleman_liau = self.calculate_coleman_liau_index()
        ari = self.calculate_automated_readability_index()
        
        # Average grade level
        grades = [flesch_kincaid, gunning_fog, smog, coleman_liau, ari]
        avg_grade = sum(grades) / len(grades)
        
        # Determine readability level
        if flesch_ease >= 70:
            level = "Easy"
        elif flesch_ease >= 50:
            level = "Standard"
        else:
            level = "Difficult"
        
        return ReadabilityScores(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_kincaid,
            gunning_fog_index=gunning_fog,
            smog_index=smog,
            coleman_liau_index=coleman_liau,
            automated_readability_index=ari,
            avg_grade_level=round(avg_grade, 2),
            readability_level=level,
        )
    
    def estimate_reading_time(self, wpm: int = 200) -> str:
        """
        Estimate reading time based on word count.
        Default: 200 words per minute (average adult reading speed)
        """
        stats = self.get_statistics()
        minutes = stats.word_count / wpm
        
        if minutes < 1:
            return "Less than 1 minute"
        elif minutes < 2:
            return "1 minute"
        else:
            return f"{int(round(minutes))} minutes"
    
    def detect_ai_patterns(self) -> Dict[str, any]:
        """
        Detect patterns commonly associated with AI-generated text.
        Returns analysis of potential AI markers.
        """
        stats = self.get_statistics()
        sentence_analysis = self.analyze_sentence_lengths()
        
        # Check for common AI transition phrases
        ai_transitions = [
            "moreover", "furthermore", "in addition", "consequently",
            "therefore", "thus", "hence", "accordingly", "as a result",
            "in conclusion", "to summarize", "in summary", "overall",
            "it is important to note", "it should be noted",
            "it is worth mentioning", "needless to say",
        ]
        
        text_lower = self.text.lower()
        transition_count = sum(1 for t in ai_transitions if t in text_lower)
        transition_ratio = transition_count / max(stats.paragraph_count, 1)
        
        # Check sentence length uniformity (AI tends to be more uniform)
        sentence_uniformity = 1 - (sentence_analysis["std_deviation"] / 
                                   max(sentence_analysis["mean"], 1))
        sentence_uniformity = max(0, min(1, sentence_uniformity))
        
        # Vocabulary diversity
        vocab_diversity = self.calculate_vocabulary_diversity()
        
        # Check for repetitive phrase patterns
        # (words repeated frequently in similar positions)
        word_freq = Counter(self.words)
        most_common = word_freq.most_common(20)
        high_freq_words = sum(1 for w, c in most_common 
                            if c > stats.word_count * 0.02 and len(w) > 4)
        
        return {
            "transition_phrase_ratio": round(transition_ratio, 2),
            "sentence_length_uniformity": round(sentence_uniformity, 2),
            "vocabulary_diversity": vocab_diversity,
            "high_frequency_word_count": high_freq_words,
            "markers_detected": transition_count,
        }
    
    def get_full_analysis(self) -> Dict[str, any]:
        """Get complete text analysis"""
        stats = self.get_statistics()
        readability = self.get_readability_scores()
        sentence_analysis = self.analyze_sentence_lengths()
        ai_patterns = self.detect_ai_patterns()
        
        return {
            "statistics": {
                "character_count": stats.character_count,
                "word_count": stats.word_count,
                "sentence_count": stats.sentence_count,
                "paragraph_count": stats.paragraph_count,
                "syllable_count": stats.syllable_count,
                "complex_word_count": stats.complex_word_count,
                "unique_word_count": stats.unique_word_count,
                "avg_word_length": stats.avg_word_length,
                "avg_sentence_length": stats.avg_sentence_length,
            },
            "readability": {
                "flesch_reading_ease": readability.flesch_reading_ease,
                "flesch_kincaid_grade": readability.flesch_kincaid_grade,
                "gunning_fog_index": readability.gunning_fog_index,
                "smog_index": readability.smog_index,
                "coleman_liau_index": readability.coleman_liau_index,
                "automated_readability_index": readability.automated_readability_index,
                "avg_grade_level": readability.avg_grade_level,
                "readability_level": readability.readability_level,
            },
            "sentence_analysis": sentence_analysis,
            "vocabulary_diversity": self.calculate_vocabulary_diversity(),
            "estimated_reading_time": self.estimate_reading_time(),
            "ai_pattern_indicators": ai_patterns,
        }


def analyze_text(text: str) -> Dict[str, any]:
    """Convenience function to analyze text"""
    analyzer = TextAnalyzer(text)
    return analyzer.get_full_analysis()
