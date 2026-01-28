#!/usr/bin/env python3
"""
Extract metrics from ginn-lp JSON outputs and codes JSON outputs for comparison.
Creates a comparison table showing metrics side-by-side.
"""

import json
import glob
import os
import argparse
from pathlib import Path

def load_ginnlp_json(json_file):
    """Load and extract metrics from ginn-lp JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    exp_info = data['experiment_info']
    dataset_name = exp_info['dataset_name']
    hyperparams = f"E{exp_info['num_epochs']}_B{exp_info['start_ln_blocks']}_G{exp_info['growth_steps']}"
    
    # For Agriculture, use original_scale metrics; for ENB, use regular metrics
    if 'metrics_original_scale' in data:
        metrics = data['metrics_original_scale']
    else:
        metrics = data.get('metrics', {})
    
    return {
        'source': 'ginn-lp',
        'dataset': dataset_name,
        'hyperparams': hyperparams,
        'MSE': metrics.get('MSE'),
        'MAE': metrics.get('MAE'),
        'RMSE': metrics.get('RMSE'),
        'MAPE': metrics.get('MAPE'),
        'equation': data.get('recovered_equation', 'N/A')
    }

def load_codes_json(json_file):
    """Load and extract metrics from codes JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    exp_info = data.get('experiment_info', {})
    initial_blocks = exp_info.get('initial_shared_blocks', '?')
    max_blocks = exp_info.get('max_blocks', '?')
    hyperparams = f"{initial_blocks}init_{max_blocks}max"
    
    # Extract metrics for each target
    nn_metrics = data.get('neural_network_metrics', {})
    
    results = []
    for target_name, metrics in nn_metrics.items():
        # Extract target identifier (Y1/Y2 or target name)
        if 'Heating' in target_name or 'Sustainability' in target_name:
            target_id = 'Y1'
        elif 'Cooling' in target_name or 'Consumer' in target_name:
            target_id = 'Y2'
        else:
            target_id = target_name
        
        results.append({
            'source': 'codes',
            'target_id': target_id,
            'target_name': target_name,
            'hyperparams': hyperparams,
            'MSE': metrics.get('MSE'),
            'MAE': metrics.get('MAE'),
            'RMSE': metrics.get('RMSE'),
            'MAPE': metrics.get('MAPE'),
            'equation': data.get('symbolic_equations', {}).get(target_id.lower().replace('y', 'target_'), 'N/A')
        })
    
    return results

def create_comparison_table(ginnlp_results, codes_results, dataset='ENB'):
    """Create a markdown comparison table"""
    
    print(f"\n{'='*80}")
    print(f"Comparison Table: {dataset} Dataset")
    print(f"{'='*80}\n")
    
    print("| Model | Target | Config | MSE | MAE | RMSE | MAPE |")
    print("|-------|--------|--------|-----|-----|------|------|")
    
    # Add codes results
    for result in codes_results:
        target_name = result['target_name'].split('(')[0].strip()
        print(f"| **codes (multi-target)** | {target_name} ({result['target_id']}) | {result['hyperparams']} | "
              f"{result['MSE']:.6f} | {result['MAE']:.6f} | {result['RMSE']:.6f} | {result['MAPE']:.4f}% |")
    
    # Add ginn-lp results
    for result in ginnlp_results:
        # Determine target from dataset name
        if 'Heating' in result['dataset'] or 'Sustainability' in result['dataset']:
            target_id = 'Y1'
            target_name = 'Heating' if 'Heating' in result['dataset'] else 'Sustainability'
        elif 'Cooling' in result['dataset'] or 'Consumer' in result['dataset']:
            target_id = 'Y2'
            target_name = 'Cooling' if 'Cooling' in result['dataset'] else 'ConsumerTrend'
        else:
            target_id = '?'
            target_name = result['dataset']
        
        print(f"| **ginn-lp (single-target)** | {target_name} ({target_id}) | {result['hyperparams']} | "
              f"{result['MSE']:.6f} | {result['MAE']:.6f} | {result['RMSE']:.6f} | {result['MAPE']:.4f}% |")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Extract and compare metrics from ginn-lp and codes JSON files')
    parser.add_argument('--ginnlp_dir', type=str, required=True, help='Directory containing ginn-lp JSON outputs')
    parser.add_argument('--codes_file', type=str, help='Path to codes JSON file (optional)')
    parser.add_argument('--dataset', type=str, choices=['ENB', 'Agriculture'], default='ENB',
                        help='Dataset type (default: ENB)')
    
    args = parser.parse_args()
    
    # Load ginn-lp results
    print(f"Loading ginn-lp results from: {args.ginnlp_dir}")
    ginnlp_files = glob.glob(os.path.join(args.ginnlp_dir, 'ginnlp_*.json'))
    
    if not ginnlp_files:
        print(f"❌ No ginn-lp JSON files found in {args.ginnlp_dir}")
        return
    
    ginnlp_results = []
    for json_file in sorted(ginnlp_files):
        try:
            result = load_ginnlp_json(json_file)
            ginnlp_results.append(result)
            print(f"✅ Loaded: {os.path.basename(json_file)}")
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    # Load codes results
    codes_results = []
    if args.codes_file and os.path.exists(args.codes_file):
        print(f"\nLoading codes results from: {args.codes_file}")
        try:
            results = load_codes_json(args.codes_file)
            codes_results.extend(results)
            print(f"✅ Loaded codes results ({len(results)} targets)")
        except Exception as e:
            print(f"⚠️  Error loading codes file: {e}")
    else:
        print(f"\n⚠️  No codes file provided. Showing only ginn-lp results.")
    
    # Create comparison table
    if ginnlp_results:
        create_comparison_table(ginnlp_results, codes_results, args.dataset)
        
        # Print detailed information
        print(f"\n{'='*80}")
        print("Detailed Results")
        print(f"{'='*80}\n")
        
        for result in ginnlp_results:
            print(f"📄 {result['dataset']} ({result['hyperparams']})")
            print(f"   MSE:  {result['MSE']:.6f}")
            print(f"   MAE:  {result['MAE']:.6f}")
            print(f"   RMSE: {result['RMSE']:.6f}")
            print(f"   MAPE: {result['MAPE']:.4f}%")
            print(f"   Equation: {result['equation'][:100]}..." if len(str(result['equation'])) > 100 else f"   Equation: {result['equation']}")
            print()
        
        if codes_results:
            print(f"\n{'='*80}")
            print("Codes Results")
            print(f"{'='*80}\n")
            
            for result in codes_results:
                print(f"📄 {result['target_name']} ({result['hyperparams']})")
                print(f"   MSE:  {result['MSE']:.6f}")
                print(f"   MAE:  {result['MAE']:.6f}")
                print(f"   RMSE: {result['RMSE']:.6f}")
                print(f"   MAPE: {result['MAPE']:.4f}%")
                print()

if __name__ == '__main__':
    main()
