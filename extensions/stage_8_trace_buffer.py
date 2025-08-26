#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8: SSRL-Integrated Trace Buffer System
===========================================

Advanced trace buffer system integrated with Self-Supervised Representation Learning (SSRL)
for intelligent trace management, representation-based sampling, and quality-aware replay.

Key Features:
- SSRL-enhanced trace representations
- Quality-based trace prioritization
- Multi-modal trace storage and retrieval
- Representation similarity-based sampling
- Integration with confidence filtering and RLHF
- Adaptive buffer management with curriculum learning

Architecture:
- Trace representation encoder using SSRL
- Quality-aware buffer management
- Similarity-based trace clustering
- Intelligent replay strategies
- Integration with other AI Research Agent stages
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import asyncio
from pathlib import Path

# Import SSRL components for integration
try:
    from extensions.stage_9_ssrl import SSRLSystem, SSRLConfig, ModalityType, RepresentationQuality
    from extensions.stage_7_confidence_filtering import ConfidenceFilterManager
    from extensions.stage_5_rlhf_agentic_rl import RewardSignal, RewardSignalType
    SSRL_AVAILABLE = True
except ImportError:
    print("⚠️ SSRL components not available. Using fallback implementations.")
    SSRL_AVAILABLE = False

class TraceType(Enum):
    """Types of traces that can be stored"""
    RESEARCH_QUERY = "research_query"
    CODE_GENERATION = "code_generation"
    REASONING_STEP = "reasoning_step"
    TOOL_USAGE = "tool_usage"
    CONTEXT_RETRIEVAL = "context_retrieval"
    RESPONSE_GENERATION = "response_generation"

class SamplingStrategy(Enum):
    """Strategies for sampling traces from buffer"""
    RANDOM = "random"
    SUCCESS_BASED = "success_based"
    CONFIDENCE_BASED = "confidence_based"
    QUALITY_BASED = "quality_based"
    SIMILARITY_BASED = "similarity_based"
    CURRICULUM_BASED = "curriculum_based"
    DIVERSITY_BASED = "diversity_based"

class BufferPriority(Enum):
    """Priority levels for trace storage"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TraceMetadata:
    """Metadata for trace entries"""
    trace_id: str
    trace_type: TraceType
    timestamp: datetime
    session_id: str
    user_id: Optional[str] = None
    priority: BufferPriority = BufferPriority.MEDIUM
    
    # Quality metrics
    success: bool = False
    confidence_score: float = 0.0
    quality_score: float = 0.0
    reward_signals: List[RewardSignal] = field(default_factory=list)
    
    # SSRL integration
    representation: Optional[torch.Tensor] = None
    representation_quality: Optional[RepresentationQuality] = None
    modality: ModalityType = ModalityType.TEXT
    
    # Context information
    context_size: int = 0
    processing_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class TraceEntry:
    """Complete trace entry with data and metadata"""
    metadata: TraceMetadata
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)
    error_info: Optional[Dict[str, Any]] = None

class SSRLTraceBuffer:
    """SSRL-integrated trace buffer with intelligent management"""
    
    def __init__(self, max_size: int = 10000, ssrl_config: Optional[SSRLConfig] = None):
        self.max_size = max_size
        self.buffer: List[TraceEntry] = []
        self.trace_index: Dict[str, int] = {}  # trace_id -> buffer index
        
        # SSRL integration
        self.ssrl_system = None
        if SSRL_AVAILABLE and ssrl_config:
            self.ssrl_system = SSRLSystem(ssrl_config)
            print("🧠 SSRL integration enabled for trace buffer")
        
        # Buffer management
        self.priority_queues = {priority: deque() for priority in BufferPriority}
        self.quality_threshold = 0.5
        self.confidence_threshold = 0.6
        
        # Statistics and monitoring
        self.buffer_stats = {
            "total_traces_added": 0,
            "traces_evicted": 0,
            "successful_traces": 0,
            "high_quality_traces": 0,
            "representation_enhanced": 0
        }
        
        # Clustering for similarity-based operations
        self.representation_clusters = defaultdict(list)
        self.cluster_centroids = {}
        
        print(f"🗃️ SSRL-Integrated Trace Buffer initialized")
        print(f"   Max size: {max_size}")
        print(f"   SSRL enabled: {self.ssrl_system is not None}")
    
    async def add_trace(self, input_data: Dict[str, Any], output_data: Dict[str, Any],
                       trace_type: TraceType, session_id: str,
                       success: bool = False, confidence_score: float = 0.0,
                       priority: BufferPriority = BufferPriority.MEDIUM,
                       **kwargs) -> str:
        """Add a trace to the buffer with SSRL enhancement"""
        
        trace_id = f"trace_{len(self.buffer)}_{datetime.now().timestamp()}"
        
        # Create metadata
        metadata = TraceMetadata(
            trace_id=trace_id,
            trace_type=trace_type,
            timestamp=datetime.now(),
            session_id=session_id,
            priority=priority,
            success=success,
            confidence_score=confidence_score,
            **kwargs
        )
        
        # Enhance with SSRL representations if available
        if self.ssrl_system:
            await self._enhance_trace_with_ssrl(metadata, input_data, output_data)
        
        # Create trace entry
        trace_entry = TraceEntry(
            metadata=metadata,
            input_data=input_data,
            output_data=output_data,
            intermediate_states=kwargs.get("intermediate_states", []),
            error_info=kwargs.get("error_info")
        )
        
        # Add to buffer with intelligent management
        self._add_to_buffer(trace_entry)
        
        # Update statistics
        self.buffer_stats["total_traces_added"] += 1
        if success:
            self.buffer_stats["successful_traces"] += 1
        if metadata.quality_score > self.quality_threshold:
            self.buffer_stats["high_quality_traces"] += 1
        
        return trace_id
    
    async def _enhance_trace_with_ssrl(self, metadata: TraceMetadata, 
                                     input_data: Dict[str, Any], 
                                     output_data: Dict[str, Any]):
        """Enhance trace with SSRL representations"""
        
        try:
            # Determine modality based on trace type and data
            modality = self._determine_modality(metadata.trace_type, input_data)
            metadata.modality = modality
            
            # Prepare data for SSRL encoding
            ssrl_data = self._prepare_ssrl_data(input_data, output_data, modality)
            
            # Get SSRL representation
            representation = self.ssrl_system.get_representations(ssrl_data, modality)
            metadata.representation = representation
            
            # Evaluate representation quality
            if hasattr(self.ssrl_system, 'quality_evaluator'):
                quality = self.ssrl_system.quality_evaluator.evaluate_representations(
                    representation.unsqueeze(0)  # Add batch dimension
                )
                metadata.representation_quality = quality
                metadata.quality_score = quality.confidence_score
            
            self.buffer_stats["representation_enhanced"] += 1
            
        except Exception as e:
            print(f"⚠️ SSRL enhancement failed for trace {metadata.trace_id}: {e}")
            # Continue without SSRL enhancement
    
    def _determine_modality(self, trace_type: TraceType, input_data: Dict[str, Any]) -> ModalityType:
        """Determine the appropriate modality for SSRL encoding"""
        
        if trace_type == TraceType.CODE_GENERATION:
            return ModalityType.CODE
        elif "graph" in input_data or "nodes" in input_data:
            return ModalityType.GRAPH
        elif "structured_data" in input_data or "features" in input_data:
            return ModalityType.STRUCTURED
        else:
            return ModalityType.TEXT
    
    def _prepare_ssrl_data(self, input_data: Dict[str, Any], 
                          output_data: Dict[str, Any], 
                          modality: ModalityType) -> Dict[str, torch.Tensor]:
        """Prepare data for SSRL encoding"""
        
        if modality == ModalityType.TEXT:
            # Combine input and output text
            text_content = f"{input_data.get('query', '')} {output_data.get('response', '')}"
            # Convert to tensor (simplified - in practice, use proper tokenization)
            return {"input_ids": torch.randn(1, 512)}  # Placeholder
        
        elif modality == ModalityType.CODE:
            code_content = input_data.get('code', '') + output_data.get('generated_code', '')
            return {"input_ids": torch.randn(1, 512)}  # Placeholder
        
        elif modality == ModalityType.GRAPH:
            return {
                "node_features": torch.randn(10, 512),  # Placeholder
                "edge_index": torch.randint(0, 10, (2, 20))  # Placeholder
            }
        
        else:  # STRUCTURED
            return {"features": torch.randn(1, 512)}  # Placeholder
    
    def _add_to_buffer(self, trace_entry: TraceEntry):
        """Add trace to buffer with intelligent management"""
        
        # Check if buffer is full
        if len(self.buffer) >= self.max_size:
            self._evict_trace()
        
        # Add to main buffer
        self.buffer.append(trace_entry)
        self.trace_index[trace_entry.metadata.trace_id] = len(self.buffer) - 1
        
        # Add to priority queue
        self.priority_queues[trace_entry.metadata.priority].append(trace_entry.metadata.trace_id)
        
        # Update representation clusters if SSRL is available
        if trace_entry.metadata.representation is not None:
            self._update_representation_clusters(trace_entry)
    
    def _evict_trace(self):
        """Evict a trace using intelligent strategy"""
        
        # Priority-based eviction: evict lowest priority, oldest traces first
        for priority in [BufferPriority.LOW, BufferPriority.MEDIUM, BufferPriority.HIGH]:
            if self.priority_queues[priority]:
                trace_id_to_evict = self.priority_queues[priority].popleft()
                self._remove_trace_by_id(trace_id_to_evict)
                self.buffer_stats["traces_evicted"] += 1
                return
        
        # If no low/medium priority traces, evict oldest high priority
        if self.priority_queues[BufferPriority.CRITICAL]:
            trace_id_to_evict = self.priority_queues[BufferPriority.CRITICAL].popleft()
            self._remove_trace_by_id(trace_id_to_evict)
            self.buffer_stats["traces_evicted"] += 1
    
    def _remove_trace_by_id(self, trace_id: str):
        """Remove trace from buffer by ID"""
        
        if trace_id in self.trace_index:
            index = self.trace_index[trace_id]
            
            # Remove from buffer
            removed_trace = self.buffer.pop(index)
            
            # Update indices for remaining traces
            for tid, idx in self.trace_index.items():
                if idx > index:
                    self.trace_index[tid] = idx - 1
            
            # Remove from index
            del self.trace_index[trace_id]
            
            # Remove from representation clusters
            if removed_trace.metadata.representation is not None:
                self._remove_from_clusters(removed_trace)
    
    def _update_representation_clusters(self, trace_entry: TraceEntry):
        """Update representation clusters for similarity-based operations"""
        
        if trace_entry.metadata.representation is None:
            return
        
        representation = trace_entry.metadata.representation
        trace_id = trace_entry.metadata.trace_id
        
        # Find closest cluster or create new one
        min_distance = float('inf')
        closest_cluster = None
        
        for cluster_id, centroid in self.cluster_centroids.items():
            distance = F.cosine_similarity(representation, centroid, dim=0).item()
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        
        # Add to closest cluster or create new one
        if closest_cluster is not None and min_distance < 0.3:  # Similarity threshold
            self.representation_clusters[closest_cluster].append(trace_id)
            # Update centroid
            cluster_representations = [
                self.get_trace_by_id(tid).metadata.representation 
                for tid in self.representation_clusters[closest_cluster]
                if self.get_trace_by_id(tid) and self.get_trace_by_id(tid).metadata.representation is not None
            ]
            if cluster_representations:
                self.cluster_centroids[closest_cluster] = torch.mean(torch.stack(cluster_representations), dim=0)
        else:
            # Create new cluster
            new_cluster_id = f"cluster_{len(self.cluster_centroids)}"
            self.representation_clusters[new_cluster_id] = [trace_id]
            self.cluster_centroids[new_cluster_id] = representation.clone()
    
    def _remove_from_clusters(self, trace_entry: TraceEntry):
        """Remove trace from representation clusters"""
        
        trace_id = trace_entry.metadata.trace_id
        
        for cluster_id, trace_ids in self.representation_clusters.items():
            if trace_id in trace_ids:
                trace_ids.remove(trace_id)
                
                # Update centroid if cluster still has traces
                if trace_ids:
                    cluster_representations = [
                        self.get_trace_by_id(tid).metadata.representation 
                        for tid in trace_ids
                        if self.get_trace_by_id(tid) and self.get_trace_by_id(tid).metadata.representation is not None
                    ]
                    if cluster_representations:
                        self.cluster_centroids[cluster_id] = torch.mean(torch.stack(cluster_representations), dim=0)
                else:
                    # Remove empty cluster
                    del self.representation_clusters[cluster_id]
                    del self.cluster_centroids[cluster_id]
                break
    
    def sample_replay_batch(self, batch_size: int = 16, 
                           strategy: SamplingStrategy = SamplingStrategy.QUALITY_BASED,
                           **kwargs) -> List[TraceEntry]:
        """Sample a batch of traces for replay using specified strategy"""
        
        if not self.buffer:
            return []
        
        if strategy == SamplingStrategy.RANDOM:
            return self._sample_random(batch_size)
        elif strategy == SamplingStrategy.SUCCESS_BASED:
            return self._sample_success_based(batch_size)
        elif strategy == SamplingStrategy.CONFIDENCE_BASED:
            return self._sample_confidence_based(batch_size, kwargs.get('confidence_threshold', self.confidence_threshold))
        elif strategy == SamplingStrategy.QUALITY_BASED:
            return self._sample_quality_based(batch_size)
        elif strategy == SamplingStrategy.SIMILARITY_BASED:
            return self._sample_similarity_based(batch_size, kwargs.get('query_representation'))
        elif strategy == SamplingStrategy.CURRICULUM_BASED:
            return self._sample_curriculum_based(batch_size, kwargs.get('difficulty_level', 0.5))
        elif strategy == SamplingStrategy.DIVERSITY_BASED:
            return self._sample_diversity_based(batch_size)
        else:
            return self._sample_random(batch_size)
    
    def _sample_random(self, batch_size: int) -> List[TraceEntry]:
        """Random sampling"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def _sample_success_based(self, batch_size: int) -> List[TraceEntry]:
        """Sample based on success rate"""
        successful_traces = [trace for trace in self.buffer if trace.metadata.success]
        if not successful_traces:
            return self._sample_random(batch_size)
        return random.sample(successful_traces, min(batch_size, len(successful_traces)))
    
    def _sample_confidence_based(self, batch_size: int, threshold: float) -> List[TraceEntry]:
        """Sample based on confidence scores"""
        confident_traces = [trace for trace in self.buffer if trace.metadata.confidence_score >= threshold]
        if not confident_traces:
            return self._sample_random(batch_size)
        return random.sample(confident_traces, min(batch_size, len(confident_traces)))
    
    def _sample_quality_based(self, batch_size: int) -> List[TraceEntry]:
        """Sample based on quality scores"""
        # Sort by quality score and sample from top traces
        sorted_traces = sorted(self.buffer, key=lambda t: t.metadata.quality_score, reverse=True)
        top_traces = sorted_traces[:min(batch_size * 2, len(sorted_traces))]  # Top 2x batch size
        return random.sample(top_traces, min(batch_size, len(top_traces)))
    
    def _sample_similarity_based(self, batch_size: int, query_representation: Optional[torch.Tensor]) -> List[TraceEntry]:
        """Sample based on similarity to query representation"""
        
        if query_representation is None or not any(t.metadata.representation is not None for t in self.buffer):
            return self._sample_random(batch_size)
        
        # Calculate similarities
        similarities = []
        for trace in self.buffer:
            if trace.metadata.representation is not None:
                similarity = F.cosine_similarity(
                    query_representation, 
                    trace.metadata.representation, 
                    dim=0
                ).item()
                similarities.append((similarity, trace))
        
        # Sort by similarity and sample
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similar = similarities[:min(batch_size * 2, len(similarities))]
        
        return random.sample([trace for _, trace in top_similar], min(batch_size, len(top_similar)))
    
    def _sample_curriculum_based(self, batch_size: int, difficulty_level: float) -> List[TraceEntry]:
        """Sample based on curriculum learning difficulty"""
        
        # Define difficulty based on quality score and success rate
        suitable_traces = []
        for trace in self.buffer:
            trace_difficulty = 1.0 - trace.metadata.quality_score  # Lower quality = higher difficulty
            if abs(trace_difficulty - difficulty_level) < 0.3:  # Within difficulty range
                suitable_traces.append(trace)
        
        if not suitable_traces:
            return self._sample_random(batch_size)
        
        return random.sample(suitable_traces, min(batch_size, len(suitable_traces)))
    
    def _sample_diversity_based(self, batch_size: int) -> List[TraceEntry]:
        """Sample for maximum diversity"""
        
        if not self.representation_clusters:
            return self._sample_random(batch_size)
        
        # Sample from different clusters
        selected_traces = []
        clusters = list(self.representation_clusters.keys())
        
        for i in range(batch_size):
            cluster_id = clusters[i % len(clusters)]
            cluster_traces = self.representation_clusters[cluster_id]
            
            if cluster_traces:
                trace_id = random.choice(cluster_traces)
                trace = self.get_trace_by_id(trace_id)
                if trace:
                    selected_traces.append(trace)
        
        return selected_traces
    
    def get_trace_by_id(self, trace_id: str) -> Optional[TraceEntry]:
        """Get trace by ID"""
        if trace_id in self.trace_index:
            index = self.trace_index[trace_id]
            return self.buffer[index]
        return None
    
    def filter_traces(self, **criteria) -> List[TraceEntry]:
        """Filter traces based on criteria"""
        
        filtered_traces = []
        
        for trace in self.buffer:
            match = True
            
            # Check each criterion
            for key, value in criteria.items():
                if key == "trace_type" and trace.metadata.trace_type != value:
                    match = False
                    break
                elif key == "success" and trace.metadata.success != value:
                    match = False
                    break
                elif key == "min_confidence" and trace.metadata.confidence_score < value:
                    match = False
                    break
                elif key == "min_quality" and trace.metadata.quality_score < value:
                    match = False
                    break
                elif key == "session_id" and trace.metadata.session_id != value:
                    match = False
                    break
                elif key == "priority" and trace.metadata.priority != value:
                    match = False
                    break
            
            if match:
                filtered_traces.append(trace)
        
        return filtered_traces
    
    def update_trace_reward(self, trace_id: str, reward_signals: List[RewardSignal]):
        """Update trace with reward signals"""
        
        trace = self.get_trace_by_id(trace_id)
        if trace:
            trace.metadata.reward_signals.extend(reward_signals)
            
            # Update quality score based on rewards
            if reward_signals:
                avg_reward = np.mean([signal.reward_value for signal in reward_signals])
                trace.metadata.quality_score = max(trace.metadata.quality_score, avg_reward)
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        
        if not self.buffer:
            return {"message": "Buffer is empty"}
        
        # Basic statistics
        stats = self.buffer_stats.copy()
        stats.update({
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "utilization": len(self.buffer) / self.max_size,
            
            # Quality statistics
            "avg_quality_score": np.mean([t.metadata.quality_score for t in self.buffer]),
            "avg_confidence_score": np.mean([t.metadata.confidence_score for t in self.buffer]),
            "success_rate": sum(1 for t in self.buffer if t.metadata.success) / len(self.buffer),
            
            # Type distribution
            "trace_type_distribution": {
                trace_type.value: sum(1 for t in self.buffer if t.metadata.trace_type == trace_type)
                for trace_type in TraceType
            },
            
            # Priority distribution
            "priority_distribution": {
                priority.name: sum(1 for t in self.buffer if t.metadata.priority == priority)
                for priority in BufferPriority
            },
            
            # SSRL statistics
            "ssrl_enabled": self.ssrl_system is not None,
            "representation_clusters": len(self.representation_clusters),
            "traces_with_representations": sum(1 for t in self.buffer if t.metadata.representation is not None)
        })
        
        return stats
    
    def save_buffer(self, filepath: str):
        """Save buffer to file"""
        
        buffer_data = {
            "metadata": {
                "max_size": self.max_size,
                "timestamp": datetime.now().isoformat(),
                "buffer_stats": self.buffer_stats
            },
            "traces": []
        }
        
        for trace in self.buffer:
            trace_data = {
                "metadata": {
                    "trace_id": trace.metadata.trace_id,
                    "trace_type": trace.metadata.trace_type.value,
                    "timestamp": trace.metadata.timestamp.isoformat(),
                    "session_id": trace.metadata.session_id,
                    "priority": trace.metadata.priority.value,
                    "success": trace.metadata.success,
                    "confidence_score": trace.metadata.confidence_score,
                    "quality_score": trace.metadata.quality_score,
                    "modality": trace.metadata.modality.value if hasattr(trace.metadata, 'modality') else "text"
                },
                "input_data": trace.input_data,
                "output_data": trace.output_data,
                "intermediate_states": trace.intermediate_states
            }
            
            # Save representation if available (convert to list for JSON serialization)
            if trace.metadata.representation is not None:
                trace_data["representation"] = trace.metadata.representation.detach().cpu().numpy().tolist()
            
            buffer_data["traces"].append(trace_data)
        
        with open(filepath, 'w') as f:
            json.dump(buffer_data, f, indent=2)
        
        print(f"💾 Buffer saved to {filepath}")
    
    def load_buffer(self, filepath: str):
        """Load buffer from file"""
        
        with open(filepath, 'r') as f:
            buffer_data = json.load(f)
        
        # Clear current buffer
        self.buffer.clear()
        self.trace_index.clear()
        self.priority_queues = {priority: deque() for priority in BufferPriority}
        
        # Load traces
        for trace_data in buffer_data["traces"]:
            metadata = TraceMetadata(
                trace_id=trace_data["metadata"]["trace_id"],
                trace_type=TraceType(trace_data["metadata"]["trace_type"]),
                timestamp=datetime.fromisoformat(trace_data["metadata"]["timestamp"]),
                session_id=trace_data["metadata"]["session_id"],
                priority=BufferPriority(trace_data["metadata"]["priority"]),
                success=trace_data["metadata"]["success"],
                confidence_score=trace_data["metadata"]["confidence_score"],
                quality_score=trace_data["metadata"]["quality_score"]
            )
            
            # Load representation if available
            if "representation" in trace_data:
                metadata.representation = torch.tensor(trace_data["representation"])
            
            trace_entry = TraceEntry(
                metadata=metadata,
                input_data=trace_data["input_data"],
                output_data=trace_data["output_data"],
                intermediate_states=trace_data.get("intermediate_states", [])
            )
            
            self.buffer.append(trace_entry)
            self.trace_index[metadata.trace_id] = len(self.buffer) - 1
            self.priority_queues[metadata.priority].append(metadata.trace_id)
        
        # Load statistics
        if "buffer_stats" in buffer_data["metadata"]:
            self.buffer_stats.update(buffer_data["metadata"]["buffer_stats"])
        
        print(f"📂 Buffer loaded from {filepath}")
        print(f"   Loaded {len(self.buffer)} traces")

# Integration function
def integrate_ssrl_trace_buffer(max_size: int = 10000, 
                               ssrl_config: Optional[SSRLConfig] = None) -> SSRLTraceBuffer:
    """Initialize SSRL-integrated trace buffer"""
    
    print("🗃️ Initializing SSRL-Integrated Trace Buffer")
    print("=" * 50)
    
    # Use default SSRL config if not provided
    if ssrl_config is None and SSRL_AVAILABLE:
        ssrl_config = SSRLConfig(
            encoder_dim=512,
            projection_dim=128,
            integrate_semantic_graph=True,
            integrate_context_engineering=True,
            integrate_confidence_filtering=True
        )
    
    # Initialize trace buffer
    trace_buffer = SSRLTraceBuffer(max_size=max_size, ssrl_config=ssrl_config)
    
    print("✅ SSRL-Integrated Trace Buffer initialized successfully!")
    print(f"   Max capacity: {max_size}")
    print(f"   SSRL integration: {'Enabled' if trace_buffer.ssrl_system else 'Disabled'}")
    print(f"   Sampling strategies: {len(SamplingStrategy)} available")
    
    return trace_buffer

# Demo function
async def demo_ssrl_trace_buffer():
    """Demonstrate SSRL trace buffer capabilities"""
    
    print("🚀 SSRL Trace Buffer Demo")
    print("=" * 40)
    
    # Initialize trace buffer
    trace_buffer = integrate_ssrl_trace_buffer(max_size=100)
    
    # Add sample traces
    print("\n📝 Adding sample traces...")
    
    sample_traces = [
        {
            "input_data": {"query": "How do neural networks work?"},
            "output_data": {"response": "Neural networks are computational models..."},
            "trace_type": TraceType.RESEARCH_QUERY,
            "session_id": "demo_session_1",
            "success": True,
            "confidence_score": 0.9,
            "priority": BufferPriority.HIGH
        },
        {
            "input_data": {"code": "def fibonacci(n):"},
            "output_data": {"generated_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
            "trace_type": TraceType.CODE_GENERATION,
            "session_id": "demo_session_2",
            "success": True,
            "confidence_score": 0.8,
            "priority": BufferPriority.MEDIUM
        },
        {
            "input_data": {"reasoning_step": "Given that A implies B..."},
            "output_data": {"conclusion": "Therefore, if A is true, B must be true"},
            "trace_type": TraceType.REASONING_STEP,
            "session_id": "demo_session_1",
            "success": False,
            "confidence_score": 0.4,
            "priority": BufferPriority.LOW
        }
    ]
    
    trace_ids = []
    for trace_data in sample_traces:
        trace_id = await trace_buffer.add_trace(**trace_data)
        trace_ids.append(trace_id)
        print(f"   Added trace: {trace_id}")
    
    # Demonstrate sampling strategies
    print(f"\n🎯 Demonstrating sampling strategies...")
    
    strategies = [
        SamplingStrategy.RANDOM,
        SamplingStrategy.SUCCESS_BASED,
        SamplingStrategy.CONFIDENCE_BASED,
        SamplingStrategy.QUALITY_BASED
    ]
    
    for strategy in strategies:
        batch = trace_buffer.sample_replay_batch(batch_size=2, strategy=strategy)
        print(f"   {strategy.value}: {len(batch)} traces sampled")
        for trace in batch:
            print(f"      - {trace.metadata.trace_id} (success: {trace.metadata.success}, confidence: {trace.metadata.confidence_score:.2f})")
    
    # Show buffer statistics
    print(f"\n📊 Buffer Statistics:")
    stats = trace_buffer.get_buffer_statistics()
    
    print(f"   Current size: {stats['current_size']}/{stats['max_size']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Avg quality score: {stats['avg_quality_score']:.3f}")
    print(f"   Avg confidence score: {stats['avg_confidence_score']:.3f}")
    print(f"   SSRL enabled: {stats['ssrl_enabled']}")
    
    # Demonstrate filtering
    print(f"\n🔍 Demonstrating trace filtering...")
    
    successful_traces = trace_buffer.filter_traces(success=True)
    high_confidence_traces = trace_buffer.filter_traces(min_confidence=0.7)
    
    print(f"   Successful traces: {len(successful_traces)}")
    print(f"   High confidence traces: {len(high_confidence_traces)}")
    
    print(f"\n✅ SSRL Trace Buffer demo completed!")
    
    return trace_buffer

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_ssrl_trace_buffer())
