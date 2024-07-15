import subprocess

def install_packages(package_list):
    for package in package_list:
        try:
            print(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}.")
            # You can choose to handle the error as per your requirement

if __name__ == "__main__":
    # List of packages to install
    packages_to_install = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow",
        "scipy",
        "adversarial-robustness-toolbox"
        "ipython",
        # Add more packages as needed
    ]

    # Install the packages
    install_packages(packages_to_install)
