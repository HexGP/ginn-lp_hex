#!/usr/bin/env python3
"""
Compare ginn-lp results with codes results for ENB dataset
"""

import json
import sys
import os

def compare_results(ginnlp_json, codes_json, target_name="Heating Load (Y1)"):
    """Compare ginn-lp and codes results"""
    
    # Load ginn-lp results
    with open(ginnlp_json) as f:
        ginnlp = json.load(f)
    
    # Load codes results
    with open(codes_json) as f:
        codes = json.load(f)
    
    print('='*80)
    print(f'COMPARISON: {target_name}')
    print('='*80)
    print()
    print('ginn-lp (single-target):')
    ginnlp_metrics = ginnlp['metrics']
    print(f'  MSE:  {ginnlp_metrics["MSE"]:.6f}')
    print(f'  MAE:  {ginnlp_metrics["MAE"]:.6f}')
    print(f'  RMSE: {ginnlp_metrics["RMSE"]:.6f}')
    print(f'  MAPE: {ginnlp_metrics["MAPE"]:.4f}%')
    print(f'  Final Blocks: {ginnlp["model_info"].get("final_blocks", "N/A")}')
    print()
    print('codes (multi-target):')
    codes_nn = codes['neural_network_metrics'][target_name]
    codes_sym = codes['symbolic_model_metrics'][target_name]
    print('  Neural Network:')
    print(f'    MSE:  {codes_nn["MSE"]:.6f}')
    print(f'    MAE:  {codes_nn["MAE"]:.6f}')
    print(f'    RMSE: {codes_nn["RMSE"]:.6f}')
    print(f'    MAPE: {codes_nn["MAPE"]:.4f}%')
    print('  Symbolic Model:')
    print(f'    MSE:  {codes_sym["MSE"]:.6f}')
    print(f'    MAE:  {codes_sym["MAE"]:.6f}')
    print(f'    RMSE: {codes_sym["RMSE"]:.6f}')
    print(f'    MAPE: {codes_sym["MAPE"]:.4f}%')
    print()
    print('='*80)
    print('ANALYSIS:')
    print('='*80)
    print(f'ginn-lp vs codes Neural Network:')
    print(f'  MSE:  {ginnlp_metrics["MSE"]:.6f} vs {codes_nn["MSE"]:.6f} ({(ginnlp_metrics["MSE"]/codes_nn["MSE"]-1)*100:+.1f}%)')
    print(f'  MAE:  {ginnlp_metrics["MAE"]:.6f} vs {codes_nn["MAE"]:.6f} ({(ginnlp_metrics["MAE"]/codes_nn["MAE"]-1)*100:+.1f}%)')
    print(f'  RMSE: {ginnlp_metrics["RMSE"]:.6f} vs {codes_nn["RMSE"]:.6f} ({(ginnlp_metrics["RMSE"]/codes_nn["RMSE"]-1)*100:+.1f}%)')
    print(f'  MAPE: {ginnlp_metrics["MAPE"]:.4f}% vs {codes_nn["MAPE"]:.4f}% ({(ginnlp_metrics["MAPE"]/codes_nn["MAPE"]-1)*100:+.1f}%)')
    print()
    print(f'ginn-lp vs codes Symbolic Model:')
    print(f'  MSE:  {ginnlp_metrics["MSE"]:.6f} vs {codes_sym["MSE"]:.6f} ({(ginnlp_metrics["MSE"]/codes_sym["MSE"]-1)*100:+.1f}%)')
    print(f'  MAE:  {ginnlp_metrics["MAE"]:.6f} vs {codes_sym["MAE"]:.6f} ({(ginnlp_metrics["MAE"]/codes_sym["MAE"]-1)*100:+.1f}%)')
    print(f'  RMSE: {ginnlp_metrics["RMSE"]:.6f} vs {codes_sym["RMSE"]:.6f} ({(ginnlp_metrics["RMSE"]/codes_sym["RMSE"]-1)*100:+.1f}%)')
    print(f'  MAPE: {ginnlp_metrics["MAPE"]:.4f}% vs {codes_sym["MAPE"]:.4f}% ({(ginnlp_metrics["MAPE"]/codes_sym["MAPE"]-1)*100:+.1f}%)')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <ginnlp_json> <codes_json> [target_name]")
        print("Example: python compare_results.py run_ENB/outputs/ginnlp_ENB_ENB2012_Y1_Heating_Load_E500_B2_G3_*.json ../codes/ENB/output/symbolic/mtr_ginn_2initial_8max_blocks_20250916.json")
        sys.exit(1)
    
    ginnlp_json = sys.argv[1]
    codes_json = sys.argv[2]
    target_name = sys.argv[3] if len(sys.argv) > 3 else "Heating Load (Y1)"
    
    compare_results(ginnlp_json, codes_json, target_name)
