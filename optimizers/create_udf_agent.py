#!/usr/bin/env python3
"""
PySpark UDF Creation Agent
Assists in creating, registering, and benchmarking PySpark UDFs
"""

import argparse
import time
from typing import Any, Callable
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, pandas_udf, col
from pyspark.sql.types import (
    StringType, IntegerType, DoubleType, BooleanType,
    ArrayType, StructType, StructField
)
import pandas as pd


class UDFAgent:
    """Agent for creating and managing PySpark UDFs"""
    
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName("UDFAgent") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def create_basic_udf(self, func: Callable = None, return_type: Any = StringType()):
        """Create a basic PySpark UDF"""
        if func is None:
            # Default example: uppercase string function
            func = lambda x: x.upper() if x else None
        
        return udf(func, return_type)
    
    def create_vectorized_udf(self, func: Callable = None, return_type: Any = StringType()):
        """Create a vectorized (Pandas) UDF for better performance"""
        if func is None:
            # Default example: uppercase string function using pandas
            @pandas_udf(return_type)
            def default_pandas_udf(s: pd.Series) -> pd.Series:
                return s.str.upper()
            return default_pandas_udf
        
        return pandas_udf(func, return_type)
    
    def create_complex_udf_examples(self):
        """Create various complex UDF examples"""
        examples = {}
        
        # 1. String manipulation UDF
        @udf(StringType())
        def extract_domain(email):
            if email and '@' in email:
                return email.split('@')[1]
            return None
        examples['extract_domain'] = extract_domain
        
        # 2. Numeric calculation UDF
        @udf(DoubleType())
        def calculate_tax(salary, tax_rate=0.2):
            if salary:
                return float(salary) * tax_rate
            return 0.0
        examples['calculate_tax'] = calculate_tax
        
        # 3. Array processing UDF
        @udf(ArrayType(StringType()))
        def split_and_clean(text, delimiter=','):
            if text:
                return [item.strip() for item in text.split(delimiter)]
            return []
        examples['split_and_clean'] = split_and_clean
        
        # 4. Struct return UDF
        schema = StructType([
            StructField("is_valid", BooleanType(), True),
            StructField("length", IntegerType(), True),
            StructField("category", StringType(), True)
        ])
        
        @udf(schema)
        def analyze_text(text):
            if not text:
                return (False, 0, "empty")
            
            length = len(text)
            if length < 10:
                category = "short"
            elif length < 50:
                category = "medium"
            else:
                category = "long"
            
            return (True, length, category)
        examples['analyze_text'] = analyze_text
        
        # 5. Vectorized aggregation UDF
        @pandas_udf(DoubleType())
        def normalized_score(scores: pd.Series) -> pd.Series:
            min_score = scores.min()
            max_score = scores.max()
            if max_score == min_score:
                return pd.Series([0.5] * len(scores))
            return (scores - min_score) / (max_score - min_score)
        examples['normalized_score'] = normalized_score
        
        return examples
    
    def benchmark_udf(self, df, udf_func, column_name, new_column_name):
        """Benchmark UDF performance"""
        start_time = time.time()
        
        # Apply UDF
        result_df = df.withColumn(new_column_name, udf_func(col(column_name)))
        
        # Force computation
        count = result_df.count()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"UDF Benchmark Results:")
        print(f"- Rows processed: {count}")
        print(f"- Execution time: {execution_time:.2f} seconds")
        print(f"- Rows per second: {count/execution_time:.0f}")
        
        return result_df, execution_time
    
    def register_udf_to_spark(self, udf_func, name: str):
        """Register UDF for use in Spark SQL"""
        self.spark.udf.register(name, udf_func)
        print(f"UDF '{name}' registered successfully for SQL queries")
        return name
    
    def compare_udf_performance(self, df, column_name):
        """Compare performance between regular and vectorized UDFs"""
        # Create test UDFs
        regular_udf = self.create_basic_udf()
        vectorized_udf = self.create_vectorized_udf()
        
        print("Comparing UDF Performance...")
        print("-" * 50)
        
        # Benchmark regular UDF
        print("\nRegular UDF:")
        _, regular_time = self.benchmark_udf(
            df, regular_udf, column_name, "regular_result"
        )
        
        # Benchmark vectorized UDF
        print("\nVectorized UDF:")
        _, vectorized_time = self.benchmark_udf(
            df, vectorized_udf, column_name, "vectorized_result"
        )
        
        # Performance comparison
        speedup = regular_time / vectorized_time
        print(f"\nPerformance Summary:")
        print(f"- Regular UDF time: {regular_time:.2f}s")
        print(f"- Vectorized UDF time: {vectorized_time:.2f}s")
        print(f"- Speedup: {speedup:.2f}x")
        
        return {
            'regular_time': regular_time,
            'vectorized_time': vectorized_time,
            'speedup': speedup
        }


def main():
    parser = argparse.ArgumentParser(description='PySpark UDF Agent')
    parser.add_argument('--task', type=str, required=True,
                        choices=['create_basic_udf', 'create_vectorized_udf', 
                                'benchmark_udf', 'register_udf_to_spark',
                                'compare_performance', 'show_examples'],
                        help='Task to perform')
    parser.add_argument('--data-size', type=int, default=100000,
                        help='Size of test data for benchmarking')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = UDFAgent()
    
    if args.task == 'show_examples':
        examples = agent.create_complex_udf_examples()
        print("Complex UDF Examples Created:")
        for name, udf_func in examples.items():
            print(f"- {name}: {udf_func}")
    
    elif args.task == 'compare_performance':
        # Create test data
        print(f"Creating test data with {args.data_size} rows...")
        test_df = agent.spark.range(0, args.data_size).withColumn(
            "text", col("id").cast(StringType())
        )
        
        # Run performance comparison
        results = agent.compare_udf_performance(test_df, "text")
    
    elif args.task == 'create_basic_udf':
        udf_func = agent.create_basic_udf()
        print("Basic UDF created successfully")
        print(f"UDF: {udf_func}")
        
        # Example usage
        test_df = agent.spark.createDataFrame([("hello",), ("world",)], ["text"])
        result = test_df.withColumn("upper_text", udf_func(col("text")))
        result.show()
    
    elif args.task == 'create_vectorized_udf':
        udf_func = agent.create_vectorized_udf()
        print("Vectorized UDF created successfully")
        print(f"UDF: {udf_func}")
        
        # Example usage
        test_df = agent.spark.createDataFrame([("hello",), ("world",)], ["text"])
        result = test_df.withColumn("upper_text", udf_func(col("text")))
        result.show()
    
    # Clean up
    agent.spark.stop()


if __name__ == "__main__":
    main()