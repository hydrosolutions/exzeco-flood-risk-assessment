# Overall Architecture

This project implements the EXZECO flood risk assessment methodology. The core logic is written in Python and organized into a `src` directory, while the main analysis workflow is orchestrated within a Jupyter Notebook located in the `notebooks` directory.

- `src/exzeco.py`: Contains the main `ExzecoAnalysis` class that implements the Monte Carlo simulation for flood risk assessment.
- `src/dem_utils.py`: Provides utilities for downloading and processing Digital Elevation Model (DEM) data.
- `src/visualization.py`: Includes the `ExzecoVisualizer` class for creating various static and interactive plots, maps, and reports from the analysis results.
- `notebooks/exzeco_pfra.ipynb`: The main notebook that demonstrates the end-to-end workflow. It's the best place to understand how the different components are used together.
- `config/config.yml`: A crucial file that defines all parameters for the analysis, such as noise levels, number of iterations, and drainage area thresholds.

# Development Workflow

1.  **Setup**: Create a conda environment using `environment.yml` and install dependencies from `requirements.txt`.
2.  **Analysis & Execution**: The primary workflow is within `notebooks/exzeco_pfra.ipynb`. When making changes to the core logic in the `src` directory, it's recommended to test them by running the relevant cells in this notebook. The notebook uses `importlib.reload()` to ensure changes in the source files are picked up without restarting the kernel.
3.  **Configuration**: Before running an analysis, review and modify `config/config.yml` to set the desired parameters. This is the standard way to control the simulation.
4.  **Testing**: The project uses `pytest`. Tests are located in the `tests/` directory (not explicitly shown in the file list, but is a standard convention). Run tests using the `pytest` command.

# Code Conventions

-   **Modularity**: The code is structured into logical modules. When adding new functionality, try to fit it into the existing modules (`dem_utils`, `exzeco`, `visualization`) or create a new one if the functionality is distinct.
-   **Configuration-Driven**: Avoid hardcoding parameters. If a parameter is likely to change between analyses, add it to `config/config.yml` and load it in the code using the `load_config` function.
-   **Visualization**: The `ExzecoVisualizer` class is the central place for all visualization tasks. When adding a new plot or map, extend this class. It's designed to work with the `results` dictionary produced by `ExzecoAnalysis`.
-   **Notebook as an Interface**: The Jupyter Notebook is the main user interface. Ensure that any new functionality is demonstrated with a clear example in the notebook.
-   **File Paths**: Use `pathlib.Path` for handling file system paths to ensure cross-platform compatibility.
