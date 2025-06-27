import os

# Base directories
base_dir = r'd:\code\ia\codesandbox\HGC-final-clean\data\Saccharomyces_cerevisiae'
source_dir = os.path.join(base_dir, 'biclusters')
dataset_name = 'collins'
target_base_dir = os.path.join(base_dir, "dynamic_ppi", dataset_name)

# Create target directories if they don't exist and copy source content
for i in range(1, 31):
    print(f"Processing target directory {i}...")
    target_folder = os.path.join(target_base_dir, f'{dataset_name}_{i}')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}")
    
    # Source file (e.g., Kkrogan14k_1.tsv)
    source_file = os.path.join(source_dir, f'{dataset_name}_{i}.tsv')

    # Destination file (network.txt)
    dest_file = os.path.join(target_folder, 'network.txt')
    
    if os.path.exists(source_file):
        # Read content from source file
        with open(source_file, 'r') as src:
            content = src.read()
        
        # Write content to destination file
        with open(dest_file, 'w') as dst:
            dst.write(content)
        
        print(f"Copied content from {source_file} to {dest_file}")
    else:
        print(f"Warning: Source file {source_file} does not exist")

print("Done copying source content to target network.txt files")
