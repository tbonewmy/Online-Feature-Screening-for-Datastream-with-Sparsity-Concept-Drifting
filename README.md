# Online-Feature-Screening-for-Datastream-with-Sparsity-Concept-Drifting

This is a Python implementation by the authors of the paper **"Online Feature Screening for Data Streams With Concept Drift"** from Dr. Mingyuan Wang and Dr. Adrian Barbu.\
Please cite this paper if you use or build on our method. [doi.org/10.1109/TKDE.2022.3232752](https://doi.org/10.1109/TKDE.2022.3232752)

## Installation

### Prerequisites

* `Python` 3.10 or newer
* `pip`
* `numpy` 2.2.4 or newer

### Note
Although the package is designed OS independent, it was only tested on Windows. 
   \
   \
**For users installing from source (e.g., if no pre-built wheels are available for your system):**
You will need a C++ compiler compatible with your Python installation:
* **Windows:** Microsoft Visual C++ Build Tools (part of Visual Studio, or standalone).
* **Linux:** `gcc` and `g++` (usually included or easily installed via your package manager, e.g., `sudo apt-get install build-essential`).
* **macOS:** Xcode Command Line Tools (install with `xcode-select --install`).

### Install via git clone
1. Clone repository
``` bash
git clone https://github.com/yourusername/repo_name.git
```
2. Navigate into the cloned repository directory
```
cd repo_name 
```
3. Install
```
pip install .
```

### Install via download
1. Download the repository
2. Unpack to your own folder your_folder/repo_name
3. Navigate into the unpacked repository directory
``` bash
cd repo_name  
```
4. Install
``` bash
pip install .
```
### Install via pip (Currently unavailable)

If pre-built wheels are available for your system on PyPI (coming soon!), you can install directly:
```
pip install pyscreeningfs
```

## Data
For .svm sparse data, visit [https://www.sysnet.ucsd.edu/projects/url/](https://www.sysnet.ucsd.edu/projects/url/) \
Download and put into `data/url_svmlight/`

## Demo
For a demo, see testing.py in the root directory.
