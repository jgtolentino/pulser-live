#!/usr/bin/env python3
"""
PySpark Plan Optimizer
Analyzes execution plans, visualizes shuffle stages, and recommends optimizations
"""

import argparse
import re
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PlanNode:
    """Represents a node in the execution plan"""
    id: int
    operator: str
    details: str
    children: List['PlanNode']
    metrics: Dict[str, str]
    

class PlanOptimizer:
    """Analyzes and optimizes Spark execution plans"""
    
    def __init__(self):
        self.shuffle_patterns = [
            'Exchange', 'Sort', 'Aggregate', 'Join', 'Window'
        ]
        self.join_patterns = {
            'BroadcastHashJoin': 'broadcast',
            'SortMergeJoin': 'sort_merge',
            'ShuffledHashJoin': 'shuffled_hash',
            'BroadcastNestedLoopJoin': 'broadcast_nested_loop',
            'CartesianProduct': 'cartesian'
        }
        
    def parse_explain_plan(self, plan_text: str) -> Dict:
        """Parse Spark explain plan text into structured format"""
        lines = plan_text.strip().split('\n')
        
        # Detect plan sections
        sections = {
            'parsed': [],
            'analyzed': [],
            'optimized': [],
            'physical': []
        }
        
        current_section = None
        for line in lines:
            if '== Parsed Logical Plan ==' in line:
                current_section = 'parsed'
            elif '== Analyzed Logical Plan ==' in line:
                current_section = 'analyzed'
            elif '== Optimized Logical Plan ==' in line:
                current_section = 'optimized'
            elif '== Physical Plan ==' in line:
                current_section = 'physical'
            elif current_section and line.strip():
                sections[current_section].append(line)
        
        # Focus on physical plan for optimization insights
        physical_plan = self._parse_physical_plan(sections['physical'])
        
        return {
            'sections': sections,
            'physical_plan': physical_plan,
            'optimizations': self._analyze_plan(physical_plan)
        }
    
    def _parse_physical_plan(self, plan_lines: List[str]) -> Dict:
        """Parse physical plan into structured format"""
        nodes = []
        shuffle_stages = []
        joins = []
        
        for i, line in enumerate(plan_lines):
            # Detect shuffle operations
            if any(pattern in line for pattern in self.shuffle_patterns):
                shuffle_stages.append({
                    'line': i,
                    'operation': line.strip(),
                    'type': self._identify_shuffle_type(line)
                })
            
            # Detect join operations
            for join_type, join_name in self.join_patterns.items():
                if join_type in line:
                    joins.append({
                        'line': i,
                        'type': join_name,
                        'operation': line.strip()
                    })
            
            nodes.append({
                'line': i,
                'indent': len(line) - len(line.lstrip()),
                'operation': line.strip()
            })
        
        return {
            'nodes': nodes,
            'shuffle_stages': shuffle_stages,
            'joins': joins,
            'total_stages': len(shuffle_stages)
        }
    
    def _identify_shuffle_type(self, line: str) -> str:
        """Identify the type of shuffle operation"""
        if 'Exchange hashpartitioning' in line:
            return 'hash_partition'
        elif 'Exchange rangepartitioning' in line:
            return 'range_partition'
        elif 'Exchange SinglePartition' in line:
            return 'single_partition'
        elif 'Sort' in line:
            return 'sort'
        elif 'Aggregate' in line:
            return 'aggregate'
        else:
            return 'other'
    
    def _analyze_plan(self, physical_plan: Dict) -> Dict:
        """Analyze plan and provide optimization recommendations"""
        recommendations = []
        warnings = []
        
        # Check for excessive shuffles
        if physical_plan['total_stages'] > 3:
            warnings.append(
                f"High number of shuffle stages detected: {physical_plan['total_stages']}. "
                "Consider caching intermediate results or restructuring transformations."
            )
        
        # Analyze joins
        for join in physical_plan['joins']:
            if join['type'] == 'cartesian':
                warnings.append(
                    "Cartesian join detected! This is extremely expensive. "
                    "Add join conditions or use broadcast join for small tables."
                )
            elif join['type'] == 'sort_merge':
                recommendations.append(
                    f"Sort-merge join at line {join['line']}. "
                    "Consider broadcast join if one table is small (<10MB)."
                )
        
        # Check for missing broadcast opportunities
        if not any(j['type'] == 'broadcast' for j in physical_plan['joins']) and physical_plan['joins']:
            recommendations.append(
                "No broadcast joins detected. Use broadcast() hint for small dimension tables."
            )
        
        # Analyze shuffle patterns
        hash_partitions = sum(
            1 for s in physical_plan['shuffle_stages'] 
            if s['type'] == 'hash_partition'
        )
        if hash_partitions > 2:
            recommendations.append(
                f"Multiple hash partitioning stages ({hash_partitions}). "
                "Consider repartitioning data once and caching."
            )
        
        return {
            'recommendations': recommendations,
            'warnings': warnings,
            'metrics': {
                'total_shuffles': physical_plan['total_stages'],
                'join_count': len(physical_plan['joins']),
                'hash_partitions': hash_partitions
            }
        }
    
    def visualize_dag(self, physical_plan: Dict) -> str:
        """Create ASCII visualization of the DAG"""
        nodes = physical_plan['nodes']
        
        visualization = ["Execution DAG:", "=" * 50]
        
        for node in nodes:
            indent = " " * (node['indent'] // 2)
            
            # Highlight shuffle stages
            is_shuffle = any(
                s['line'] == node['line'] 
                for s in physical_plan['shuffle_stages']
            )
            
            # Highlight joins
            join_info = next(
                (j for j in physical_plan['joins'] if j['line'] == node['line']), 
                None
            )
            
            prefix = ""
            if is_shuffle:
                prefix = "[SHUFFLE] "
            if join_info:
                prefix = f"[JOIN:{join_info['type'].upper()}] "
            
            visualization.append(f"{indent}{prefix}{node['operation']}")
        
        visualization.append("=" * 50)
        visualization.append(f"Total Shuffle Stages: {physical_plan['total_stages']}")
        
        return "\n".join(visualization)
    
    def detect_skew(self, plan_text: str, stats: Optional[Dict] = None) -> Dict:
        """Detect potential data skew issues"""
        skew_indicators = []
        
        # Look for skew-related patterns in plan
        if 'skew' in plan_text.lower():
            skew_indicators.append("Explicit skew handling detected in plan")
        
        # Check for uneven partition sizes if stats provided
        if stats and 'partition_sizes' in stats:
            sizes = stats['partition_sizes']
            avg_size = sum(sizes) / len(sizes)
            max_size = max(sizes)
            
            if max_size > avg_size * 2:
                skew_ratio = max_size / avg_size
                skew_indicators.append(
                    f"Data skew detected: largest partition is {skew_ratio:.1f}x average"
                )
        
        # Look for signs of skew in plan
        if 'SortMergeJoin' in plan_text and 'skewJoin' not in plan_text:
            skew_indicators.append(
                "Sort-merge join without skew handling. Enable adaptive skew join."
            )
        
        return {
            'has_skew': len(skew_indicators) > 0,
            'indicators': skew_indicators,
            'recommendations': [
                "Enable spark.sql.adaptive.skewJoin.enabled",
                "Consider salting keys for skewed joins",
                "Use broadcast join for small tables to avoid skew issues"
            ] if skew_indicators else []
        }
    
    def recommend_broadcast_joins(self, plan_text: str, table_sizes: Optional[Dict] = None) -> List[str]:
        """Recommend tables suitable for broadcast joins"""
        recommendations = []
        
        # Default broadcast threshold (10MB)
        broadcast_threshold = 10 * 1024 * 1024
        
        if table_sizes:
            for table, size in table_sizes.items():
                if size < broadcast_threshold and f"broadcast({table})" not in plan_text:
                    size_mb = size / (1024 * 1024)
                    recommendations.append(
                        f"Table '{table}' ({size_mb:.1f}MB) is small enough for broadcast join. "
                        f"Use: broadcast({table})"
                    )
        
        # Generic recommendations based on plan analysis
        if 'SortMergeJoin' in plan_text:
            recommendations.append(
                "Sort-merge join detected. Consider using broadcast join for dimension tables."
            )
        
        return recommendations
    
    def generate_optimization_report(self, plan_text: str, 
                                   stats: Optional[Dict] = None,
                                   table_sizes: Optional[Dict] = None) -> str:
        """Generate comprehensive optimization report"""
        # Parse plan
        parsed = self.parse_explain_plan(plan_text)
        
        # Analyze components
        skew_analysis = self.detect_skew(plan_text, stats)
        broadcast_recommendations = self.recommend_broadcast_joins(plan_text, table_sizes)
        
        # Build report
        report = [
            "Spark Execution Plan Optimization Report",
            "=" * 50,
            "",
            "## Plan Analysis",
            self.visualize_dag(parsed['physical_plan']),
            "",
            "## Optimization Recommendations",
            "-" * 30
        ]
        
        # Add warnings
        if parsed['optimizations']['warnings']:
            report.append("\n### ⚠️  Warnings:")
            for warning in parsed['optimizations']['warnings']:
                report.append(f"- {warning}")
        
        # Add recommendations
        if parsed['optimizations']['recommendations']:
            report.append("\n### 💡 Recommendations:")
            for rec in parsed['optimizations']['recommendations']:
                report.append(f"- {rec}")
        
        # Add broadcast join recommendations
        if broadcast_recommendations:
            report.append("\n### 📡 Broadcast Join Opportunities:")
            for rec in broadcast_recommendations:
                report.append(f"- {rec}")
        
        # Add skew analysis
        if skew_analysis['has_skew']:
            report.append("\n### 📊 Data Skew Analysis:")
            for indicator in skew_analysis['indicators']:
                report.append(f"- {indicator}")
            report.append("\nSkew Handling Recommendations:")
            for rec in skew_analysis['recommendations']:
                report.append(f"- {rec}")
        
        # Add configuration suggestions
        report.extend([
            "",
            "## Recommended Spark Configurations",
            "-" * 30,
            "spark.sql.adaptive.enabled=true",
            "spark.sql.adaptive.coalescePartitions.enabled=true",
            "spark.sql.adaptive.skewJoin.enabled=true",
            "spark.sql.autoBroadcastJoinThreshold=10MB",
            "spark.sql.shuffle.partitions=200  # Adjust based on data size"
        ])
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='PySpark Plan Optimizer')
    parser.add_argument('--explain', type=str, required=True,
                        help='Path to explain plan text file')
    parser.add_argument('--stats', type=str,
                        help='Path to JSON file with partition statistics')
    parser.add_argument('--table-sizes', type=str,
                        help='Path to JSON file with table sizes')
    parser.add_argument('--output', type=str,
                        help='Output file for optimization report')
    
    args = parser.parse_args()
    
    # Read explain plan
    with open(args.explain, 'r') as f:
        plan_text = f.read()
    
    # Read optional stats
    stats = None
    if args.stats:
        with open(args.stats, 'r') as f:
            stats = json.load(f)
    
    # Read optional table sizes
    table_sizes = None
    if args.table_sizes:
        with open(args.table_sizes, 'r') as f:
            table_sizes = json.load(f)
    
    # Create optimizer and generate report
    optimizer = PlanOptimizer()
    report = optimizer.generate_optimization_report(plan_text, stats, table_sizes)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Optimization report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()