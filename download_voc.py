"""
Download and extract PASCAL VOC 2007 dataset.
"""
import os
import urllib.request
import tarfile
from pathlib import Path
import shutil

# PASCAL VOC 2007 download URLs
# Note: If direct download fails, you may need to download manually from:
# http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
VOC2007_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC2007_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

# Alternative: Try using HTTPS
VOC2007_URL_HTTPS = "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar"
VOC2007_TEST_URL_HTTPS = "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar"

DATA_DIR = Path("data")
VOC_DIR = DATA_DIR / "voc2007"


def is_valid_tar(tar_path: Path) -> bool:
    """Check if tar file is valid."""
    if not tar_path.exists():
        return False
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.getmembers()  # Try to read members
        return True
    except (tarfile.ReadError, tarfile.TarError, IOError):
        return False


def download_file(url: str, dest_path: Path, force_redownload: bool = False, min_size_mb: float = 10.0):
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")
    print(f"Destination: {dest_path}")
    
    # Check if file exists and is valid
    if dest_path.exists() and not force_redownload:
        if dest_path.suffix == '.tar':
            if is_valid_tar(dest_path):
                file_size_mb = dest_path.stat().st_size / (1024 * 1024)
                if file_size_mb >= min_size_mb:
                    print(f"File already exists and is valid: {dest_path} ({file_size_mb:.2f} MB)")
                    return
                else:
                    print(f"File exists but is too small ({file_size_mb:.2f} MB). Re-downloading...")
                    dest_path.unlink()
            else:
                print(f"File exists but is corrupted. Re-downloading...")
                dest_path.unlink()
        else:
            print(f"File already exists: {dest_path}")
            return
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create request with proper headers
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            if total_size > 0:
                print(f"File size: {total_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                    else:
                        print(f"\rDownloaded: {downloaded} bytes", end="")
        
        print(f"\nDownloaded: {dest_path}")
        
        # Check file size
        file_size_mb = dest_path.stat().st_size / (1024 * 1024)
        if file_size_mb < min_size_mb:
            print(f"Warning: Downloaded file is too small ({file_size_mb:.2f} MB). It may be corrupted.")
            dest_path.unlink()
            raise ValueError(f"Downloaded file is too small ({file_size_mb:.2f} MB, expected at least {min_size_mb} MB)")
        
        # Validate downloaded tar file
        if dest_path.suffix == '.tar' and not is_valid_tar(dest_path):
            print(f"Warning: Downloaded file appears corrupted. Please try again.")
            dest_path.unlink()
            raise ValueError("Downloaded tar file is corrupted")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise


# download_progress function removed - now handled in download_file


def extract_tar(tar_path: Path, extract_dir: Path):
    """Extract tar file to directory."""
    print(f"\nExtracting {tar_path.name}...")
    
    # Validate tar file before extraction
    if not is_valid_tar(tar_path):
        raise ValueError(f"Tar file is corrupted or invalid: {tar_path}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        raise


def organize_voc_structure():
    """Organize VOC dataset into proper structure."""
    # Find the VOCdevkit directory
    vocdevkit = DATA_DIR / "VOCdevkit"
    if not vocdevkit.exists():
        # Check if it's already in the right place
        if (DATA_DIR / "VOC2007").exists():
            vocdevkit = DATA_DIR / "VOC2007"
        else:
            print("Warning: Could not find VOCdevkit or VOC2007 directory")
            return
    
    # Move to voc2007 directory
    if vocdevkit.name == "VOCdevkit":
        voc2007_source = vocdevkit / "VOC2007"
        if voc2007_source.exists():
            if VOC_DIR.exists():
                shutil.rmtree(VOC_DIR)
            shutil.move(str(voc2007_source), str(VOC_DIR))
            print(f"Organized dataset to: {VOC_DIR}")
    
    # Clean up
    if (DATA_DIR / "VOCdevkit").exists():
        shutil.rmtree(DATA_DIR / "VOCdevkit")


def main():
    """Main function to download and extract VOC 2007."""
    print("=" * 60)
    print("PASCAL VOC 2007 Dataset Downloader")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if VOC_DIR.exists() and (VOC_DIR / "Annotations").exists() and (VOC_DIR / "JPEGImages").exists():
        print(f"\nDataset already exists at: {VOC_DIR}")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Download trainval set (expected size ~430 MB)
    trainval_tar = DATA_DIR / "VOCtrainval_06-Nov-2007.tar"
    # Check if tar is corrupted and needs re-download
    if trainval_tar.exists() and not is_valid_tar(trainval_tar):
        print("Existing trainval tar file is corrupted. Re-downloading...")
        trainval_tar.unlink()
    
    # Try downloading
    download_success = False
    try:
        print("Trying primary download URL...")
        download_file(VOC2007_URL, trainval_tar, min_size_mb=100.0)
        download_success = True
    except Exception as e:
        print(f"Primary URL failed: {e}")
        print("\n" + "="*60)
        print("AUTOMATIC DOWNLOAD FAILED")
        print("="*60)
        print("\nPlease download the dataset manually:")
        print(f"1. Visit: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/")
        print(f"2. Download: VOCtrainval_06-Nov-2007.tar")
        print(f"3. Save it to: {trainval_tar.absolute()}")
        print(f"\nAlternatively, you can use torchvision to download:")
        print("   from torchvision.datasets import VOCDetection")
        print("   dataset = VOCDetection(root='data', year='2007', download=True)")
        print("\n" + "="*60)
        raise
    
    if download_success:
        extract_tar(trainval_tar, DATA_DIR)
    
    # Download test set (expected size ~430 MB)
    test_tar = DATA_DIR / "VOCtest_06-Nov-2007.tar"
    # Check if tar is corrupted and needs re-download
    if test_tar.exists() and not is_valid_tar(test_tar):
        print("Existing test tar file is corrupted. Re-downloading...")
        test_tar.unlink()
    
    # Try downloading
    download_success = False
    try:
        print("Trying primary download URL...")
        download_file(VOC2007_TEST_URL, test_tar, min_size_mb=100.0)
        download_success = True
    except Exception as e:
        print(f"Primary URL failed: {e}")
        print("\n" + "="*60)
        print("AUTOMATIC DOWNLOAD FAILED")
        print("="*60)
        print("\nPlease download the dataset manually:")
        print(f"1. Visit: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/")
        print(f"2. Download: VOCtest_06-Nov-2007.tar")
        print(f"3. Save it to: {test_tar.absolute()}")
        print("\n" + "="*60)
        raise
    
    if download_success:
        extract_tar(test_tar, DATA_DIR)
    
    # Organize structure
    organize_voc_structure()
    
    # Clean up tar files
    print("\nCleaning up tar files...")
    if trainval_tar.exists():
        trainval_tar.unlink()
    if test_tar.exists():
        test_tar.unlink()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Dataset location: {VOC_DIR}")
    print("=" * 60)
    
    # Verify structure
    required_dirs = ["Annotations", "JPEGImages", "ImageSets"]
    missing = [d for d in required_dirs if not (VOC_DIR / d).exists()]
    if missing:
        print(f"Warning: Missing directories: {missing}")
    else:
        print("\nDataset structure verified successfully!")


if __name__ == "__main__":
    main()

