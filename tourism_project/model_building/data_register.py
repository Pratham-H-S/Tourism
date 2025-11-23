from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
import os

print("="*80)
print("HUGGING FACE DATA REGISTRATION")
print("="*80)

# Get token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Error: HF_TOKEN environment variable not set")
    print("\nTo fix this:")
    print("1. Get your token from: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' access")
    print("3. Set it in Colab:")
    print("   import os")
    print("   os.environ['HF_TOKEN'] = 'hf_your_token_here'")
    exit(1)

# Configuration
USERNAME = "DD009"  # Your HuggingFace username
DATASET_NAME = "Tourism"
repo_id = f"{USERNAME}/{DATASET_NAME}"
repo_type = "dataset"

print(f"\nRepository ID: {repo_id}")
print(f"Repository Type: {repo_type}")

try:
    # Login first
    print("\nStep 1: Logging in to Hugging Face...")
    login(token=HF_TOKEN, add_to_git_credential=True)
    print("✓ Successfully logged in!")
    
    # Initialize API client
    api = HfApi(token=HF_TOKEN)
    
    # Step 2: Check if repository exists
    print(f"\nStep 2: Checking if repository exists...")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✓ Repository '{repo_id}' already exists")
        print(f"  URL: https://huggingface.co/datasets/{repo_id}")
        
    except RepositoryNotFoundError:
        # Repository doesn't exist, create it
        print(f"Repository '{repo_id}' not found. Creating new repository...")
        
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False,
            token=HF_TOKEN,
            exist_ok=True
        )
        
        print(f"✓ Repository '{repo_id}' created successfully!")
        print(f"  URL: https://huggingface.co/datasets/{repo_id}")
    
    # Step 3: Upload data folder
    print(f"\nStep 3: Uploading data folder...")
    
    # Check if folder exists
    folder_path = "tourism_project/data"
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: Folder '{folder_path}' not found")
        print(f"Creating folder and checking for data files...")
        os.makedirs(folder_path, exist_ok=True)
        
        # Check for data files in current directory
        import glob
        csv_files = glob.glob("*.csv")
        if csv_files:
            print(f"Found CSV files: {csv_files}")
            # Copy first CSV to data folder
            import shutil
            for csv_file in csv_files:
                if 'tourism' in csv_file.lower():
                    dest = os.path.join(folder_path, csv_file)
                    shutil.copy(csv_file, dest)
                    print(f"✓ Copied {csv_file} to {folder_path}")
                    break
    
    # List files to upload
    import glob
    files_to_upload = glob.glob(os.path.join(folder_path, "*"))
    print(f"\nFiles to upload from '{folder_path}':")
    for file in files_to_upload:
        print(f"  - {os.path.basename(file)}")
    
    if not files_to_upload:
        print("⚠️  Warning: No files found to upload")
        print("\nPlease ensure your data files are in the 'tourism_project/data' folder")
        print("Or run data preparation first to generate the files")
    else:
        # Upload folder
        print(f"\nUploading folder to Hugging Face...")
        
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN
        )
        
        print(f"\n✓ Data uploaded successfully!")
        print(f"📊 View your dataset: https://huggingface.co/datasets/{repo_id}")
    
    print("\n" + "="*80)
    print("DATA REGISTRATION COMPLETE!")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n" + "="*80)
    print("TROUBLESHOOTING GUIDE")
    print("="*80)
    print("\n1. Token Issues:")
    print("   - Verify your token has 'write' permissions")
    print("   - Get new token: https://huggingface.co/settings/tokens")
    print("   - Make sure token starts with 'hf_'")
    
    print("\n2. Repository Issues:")
    print("   - Check repository name doesn't have special characters")
    print("   - Verify username is correct")
    print("   - Try creating repository manually first:")
    print(f"     https://huggingface.co/new-dataset?name={DATASET_NAME}")
    
    print("\n3. Data Issues:")
    print("   - Ensure data files exist in 'tourism_project/data' folder")
    print("   - Run data preparation script first")
    print("   - Check file permissions")
    
    print("\n4. Network Issues:")
    print("   - Check internet connection")
    print("   - Try again in a few minutes")
    print("   - Verify not behind firewall blocking HuggingFace")
    
    raise  # Re-raise the exception for debugging
