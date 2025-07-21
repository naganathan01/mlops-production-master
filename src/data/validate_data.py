# src/data/validate_data.py
import argparse
import json
from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger()

class DataValidator:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive data quality checks"""
        results = {
            'data_shape': df.shape,
            'missing_values': self._check_missing_values(df),
            'duplicates': self._check_duplicates(df),
            'outliers': self._detect_outliers(df),
            'data_types': self._validate_data_types(df),
            'feature_distribution': self._analyze_distributions(df),
            'correlation_issues': self._check_correlations(df),
            'target_distribution': self._analyze_target_distribution(df)
        }
        
        # Overall pass/fail
        results['data_quality'] = {
            'passed': self._evaluate_overall_quality(results),
            'issues_found': self._count_issues(results)
        }
        
        return results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'critical_missing': missing_percentages[missing_percentages > 50].to_dict()
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows"""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        return {
            'duplicate_rows': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage),
            'has_duplicates': duplicate_count > 0
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_columns:
            if col == 'target':
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        return outlier_info
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict:
        """Validate data types"""
        type_info = {}
        for col in df.columns:
            type_info[col] = {
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'non_null_count': int(df[col].count())
            }
        
        return type_info
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """Analyze feature distributions"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        distribution_info = {}
        
        for col in numeric_columns:
            if col == 'target':
                continue
                
            values = df[col].dropna()
            
            # Test for normality
            _, p_value = stats.normaltest(values)
            
            distribution_info[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values)),
                'is_normal': p_value > 0.05,
                'normality_p_value': float(p_value)
            }
        
        return distribution_info
    
    def _check_correlations(self, df: pd.DataFrame) -> Dict:
        """Check for high correlations between features"""
        numeric_df = df.select_dtypes(include=[np.number])
        if 'target' in numeric_df.columns:
            feature_df = numeric_df.drop('target', axis=1)
        else:
            feature_df = numeric_df
        
        correlation_matrix = feature_df.corr()
        
        # Find high correlations (above 0.9)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > 0.9:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'high_correlations': high_correlations,
            'max_correlation': float(correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].max()) if len(correlation_matrix) > 1 else 0
        }
    
    def _analyze_target_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze target variable distribution"""
        if 'target' not in df.columns:
            return {'error': 'No target column found'}
        
        target = df['target']
        value_counts = target.value_counts()
        
        return {
            'unique_values': int(target.nunique()),
            'value_counts': value_counts.to_dict(),
            'is_balanced': self._check_class_balance(value_counts),
            'missing_target': int(target.isnull().sum())
        }
    
    def _check_class_balance(self, value_counts: pd.Series) -> bool:
        """Check if classes are reasonably balanced"""
        if len(value_counts) < 2:
            return False
        
        min_count = value_counts.min()
        max_count = value_counts.max()
        ratio = min_count / max_count
        
        return ratio > 0.1  # Classes are balanced if minority class is at least 10%
    
    def _evaluate_overall_quality(self, results: Dict) -> bool:
        """Evaluate overall data quality"""
        # Check critical issues
        critical_missing = len(results['missing_values']['critical_missing'])
        high_duplicate_rate = results['duplicates']['duplicate_percentage'] > 20
        target_issues = 'error' in results['target_distribution']
        
        # Data quality passes if no critical issues
        return critical_missing == 0 and not high_duplicate_rate and not target_issues
    
    def _count_issues(self, results: Dict) -> int:
        """Count total number of data quality issues"""
        issues = 0
        
        # Missing value issues
        issues += len(results['missing_values']['critical_missing'])
        
        # Duplicate issues
        if results['duplicates']['duplicate_percentage'] > 20:
            issues += 1
        
        # High correlation issues
        issues += len(results['correlation_issues']['high_correlations'])
        
        # Target distribution issues
        if not results['target_distribution'].get('is_balanced', True):
            issues += 1
        
        return issues

def main():
    parser = argparse.ArgumentParser(description='Validate training data quality')
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--output', required=True, help='Output validation report JSON')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.data_path)
        logger.info("Data loaded successfully", shape=df.shape)
    except Exception as e:
        logger.error("Failed to load data", error=str(e))
        exit(1)
    
    # Run validation
    validator = DataValidator()
    results = validator.validate_data_quality(df)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("=== Data Validation Report ===")
    print(f"Data Shape: {results['data_shape']}")
    print(f"Missing Values: {results['missing_values']['total_missing']}")
    print(f"Duplicate Rows: {results['duplicates']['duplicate_rows']}")
    print(f"Data Quality: {'✅ PASSED' if results['data_quality']['passed'] else '❌ FAILED'}")
    print(f"Issues Found: {results['data_quality']['issues_found']}")
    
    # Exit with appropriate code
    exit(0 if results['data_quality']['passed'] else 1)

if __name__ == "__main__":
    main()