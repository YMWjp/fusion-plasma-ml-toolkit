#!/usr/bin/env python3
"""
テストデータセットアップツール

複数のテストファイルをraw_dataディレクトリに配置して
バッチ処理のテストを可能にする
"""

import os
import shutil
import sys

def setup_test_data():
    """テスト用のデータファイルをコピーして複数ファイルのテスト環境を作成"""
    
    raw_data_dir = "raw_data"
    source_file = os.path.join(raw_data_dir, "DivIis_tor_sum@163402_1.txt")
    
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False
    
    # テスト用のファイル名リスト（異なるshot番号を模擬）
    test_files = [
        "DivIis_tor_sum@163402_1.txt",  # 元ファイル
        "DivIis_tor_sum@163403_1.txt",  # コピー1
        "DivIis_tor_sum@163404_1.txt",  # コピー2
        "DivIis_tor_sum@163405_1.txt",  # コピー3
        "DivIis_tor_sum@163406_1.txt",  # コピー4
    ]
    
    print(f"🔧 Setting up test data in {raw_data_dir}...")
    
    for i, filename in enumerate(test_files):
        target_path = os.path.join(raw_data_dir, filename)
        
        if i == 0:
            # 元ファイルはそのまま
            if os.path.exists(target_path):
                print(f"  ✓ {filename} (original)")
            else:
                print(f"  ❌ {filename} (missing original)")
        else:
            # コピーを作成
            try:
                shutil.copy2(source_file, target_path)
                print(f"  ✓ {filename} (copied from original)")
            except Exception as e:
                print(f"  ❌ {filename} (copy failed: {e})")
    
    # 結果を確認
    actual_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.txt')]
    print(f"\n📁 Current files in {raw_data_dir}: {len(actual_files)} files")
    for f in sorted(actual_files):
        size_kb = os.path.getsize(os.path.join(raw_data_dir, f)) / 1024
        print(f"    - {f} ({size_kb:.1f} KB)")
    
    print(f"\n✅ Test data setup completed!")
    print(f"💡 You can now run: python batch_analysis.py")
    
    return True

def clean_test_data():
    """テストデータをクリーンアップ"""
    raw_data_dir = "raw_data"
    original_file = "DivIis_tor_sum@163402_1.txt"
    
    print(f"🧹 Cleaning test data in {raw_data_dir}...")
    
    files_to_remove = []
    if os.path.exists(raw_data_dir):
        for filename in os.listdir(raw_data_dir):
            if filename.endswith('.txt') and filename != original_file:
                files_to_remove.append(filename)
    
    for filename in files_to_remove:
        file_path = os.path.join(raw_data_dir, filename)
        try:
            os.remove(file_path)
            print(f"  🗑️  Removed: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to remove {filename}: {e}")
    
    print(f"✅ Cleanup completed! Kept original: {original_file}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup or clean test data for batch analysis')
    parser.add_argument('action', choices=['setup', 'clean'], 
                       help='Action to perform: setup test data or clean it')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        success = setup_test_data()
    elif args.action == 'clean':
        clean_test_data()
        success = True
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()