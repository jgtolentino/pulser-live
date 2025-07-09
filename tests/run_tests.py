#!/usr/bin/env python3
"""
Test runner for JamPacked Creative Intelligence Suite
Runs all unit and integration tests with coverage reporting
"""

import sys
import unittest
import asyncio
from pathlib import Path
import coverage
import json
import datetime

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all tests with coverage reporting"""
    
    print("🧪 JamPacked Creative Intelligence Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.datetime.now()}")
    print()
    
    # Initialize coverage
    cov = coverage.Coverage(source=[
        str(project_root / 'autonomous-intelligence'),
        str(project_root / 'engines'),
        str(project_root / 'config')
    ])
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Generate coverage report
    print("\n" + "=" * 60)
    print("📊 Coverage Report")
    print("=" * 60)
    cov.report()
    
    # Generate HTML coverage report
    html_dir = project_root / 'coverage_html'
    cov.html_report(directory=str(html_dir))
    print(f"\n📄 HTML coverage report generated at: {html_dir}")
    
    # Generate JSON report for CI/CD
    json_report = project_root / 'coverage.json'
    cov.json_report(outfile=str(json_report))
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


def run_specific_test(test_module=None, test_class=None, test_method=None):
    """Run specific test module, class, or method"""
    
    if test_module:
        # Run specific module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_module)
    elif test_class:
        # Run specific class
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_class)
    elif test_method:
        # Run specific method
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_method)
    else:
        print("No specific test specified")
        return 1
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def run_async_tests():
    """Run async test cases"""
    
    print("🔄 Running async tests...")
    
    # Import async test modules
    from test_jampacked_integration import (
        TestJamPackedIntegration,
        TestAutonomousCapabilities,
        TestMCPIntegrationEndToEnd
    )
    
    # Create async test suite
    suite = unittest.TestSuite()
    
    # Add async test methods
    async_tests = [
        TestJamPackedIntegration('test_analyze_and_store'),
        TestAutonomousCapabilities('test_pattern_discovery'),
        TestAutonomousCapabilities('test_meta_learning'),
        TestAutonomousCapabilities('test_causal_discovery'),
        TestMCPIntegrationEndToEnd('test_full_workflow')
    ]
    
    for test in async_tests:
        suite.addTest(test)
    
    # Run with async support
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def generate_test_report():
    """Generate comprehensive test report"""
    
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'project': 'JamPacked Creative Intelligence',
        'test_categories': {
            'unit_tests': {
                'creative_intelligence': 'Tests for attention, emotion, brand recall',
                'multimodal_analysis': 'Tests for cross-modal fusion and language support',
                'pattern_discovery': 'Tests for novel and causal pattern detection',
                'cultural_analysis': 'Tests for cultural appropriateness and adaptation'
            },
            'integration_tests': {
                'mcp_integration': 'Tests for SQLite MCP server integration',
                'autonomous_capabilities': 'Tests for evolutionary and meta-learning',
                'end_to_end': 'Tests for complete workflow from analysis to query'
            }
        },
        'coverage_targets': {
            'minimum': 80,
            'target': 90,
            'critical_modules': [
                'jampacked_custom_intelligence',
                'jampacked_sqlite_integration',
                'autonomous_jampacked',
                'evolutionary_learning'
            ]
        }
    }
    
    report_path = project_root / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Test report generated at: {report_path}")


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--async':
            # Run only async tests
            result = run_async_tests()
            sys.exit(0 if result.wasSuccessful() else 1)
        elif sys.argv[1] == '--report':
            # Generate test report
            generate_test_report()
            sys.exit(0)
        elif sys.argv[1].startswith('test_'):
            # Run specific test
            exit_code = run_specific_test(test_module=sys.argv[1])
            sys.exit(exit_code)
    
    # Run all tests by default
    exit_code = run_all_tests()
    
    # Generate report after tests
    generate_test_report()
    
    sys.exit(exit_code)