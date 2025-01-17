

# Creating graphs and tables readme.


This repository includes tools and scripts for generating graphs and tables to visualize and analyze data. The primary focus is on creating workflow graphs and detailed tables that represent the information that is shared between server and centers.


**Workflow Graphs**: Generate visual representations of workflows using Graphviz, illustrating the connections between shared states and function blocks.

**Tables**: Create detailed tables that summarize the data that is shared between
server and center.

## Installation

### 1. Install graphviz

On macOS:

```
brew install graphviz
```

On Ubuntu:

```
sudo apt-get update
sudo apt-get install graphviz
```




### 5. Set up your `paths.yaml` file

At the root of `fedeca_graphs`, you will find a `workflow.txt` file and
a `paths.yaml` file. The `workflow.txt` file is a raw file generated
by a run of fedeca which records all the shared states.

You must modify the `paths.yaml` with the paths of your choice:

1. `raw_workflow_file`:
This field specifies the path to the raw workflow log file. This file contains the initial, unprocessed workflow data that needs to be cleaned and analyzed.
Please point to the `workflow.txt` file.


2. `cleaned_workflow_file`:
This field specifies the path to the cleaned workflow log file. This file will be generated after processing the raw workflow file, removing duplicate `<remote_data>` blocks, and merging consecutive `<iteration>` blocks.

3. `processed_workflow_dir`:
This field specifies the directory where processed workflow files will be stored. This directory can contain various intermediate and final processed files related to the workflow.

4. `output_dir`:
This field specifies the directory where output files will be stored. This directory is used to store the final outputs generated from the workflow processing, such as graphs, tables, and other analysis results.

## Usage

### Step 1. How to Use the Log File Cleaning Script


This script is designed to clean a raw workflow log file by filtering out duplicate `<remote_data>` blocks and merging consecutive `<iteration>` blocks with the same content. The cleaned log file is then saved to a specified output path.


#### Steps to Use the Script

1. **Prepare the `paths.yaml` File**:
   - Update the `paths.yaml` file with the paths to your raw and cleaned workflow files. Here is an example:
     ```yaml
     raw_workflow_file: "/path/to/your/raw/workflow.txt"
     cleaned_workflow_file: "/path/to/your/cleaned/workflow.txt"
     ```

2. **Run the Script**:
   - Execute the script to clean the raw workflow log file. The script will read the paths from the `paths.yaml` file, process the log file, and save the cleaned log file to the specified output path.
   - You can run the script using the following command:
     ```sh
     python clean_log_file.py
     ```

#### Script Overview

1. **merge_iterations(input_text: str) -> str**:
   - Merges consecutive `<iteration>` blocks with the same content, combining their numbers.
   - **Parameters**: `input_text` (str) - The input text containing `<iteration>` blocks.
   - **Returns**: The input text with merged `<iteration>` blocks.

2. **filter_remote_data(input_text: str) -> str**:
   - Filters out duplicate `<remote_data>` blocks with the same content.
   - **Parameters**: `input_text` (str) - The input text containing `<remote_data>` blocks.
   - **Returns**: The input text with duplicate `<remote_data>` blocks removed.

3. **process_log_file(input_path: pathlib.Path, output_path: pathlib.Path) -> None**:
   - Processes a log file by filtering `<remote_data>` blocks and merging `<iteration>` blocks.
   - **Parameters**:
     - `input_path` (pathlib.Path) - The path to the input log file.
     - `output_path` (pathlib.Path) - The path to the output log file.

4. **main()**:
   - Cleans the raw workflow log file by loading paths from a YAML file, processing the log file, and writing the cleaned log file to the specified output path.

By following these steps, you can effectively clean your raw workflow log file, making it more concise and easier to analyze.


### Step 2. How to Use the Workflow Tree Creation Script


This script processes a cleaned workflow log file to create a tree representation of the workflow. It extracts shared states, remote functions, iteration blocks, and function blocks, and saves them as pickle files for further analysis.


#### Steps to Use the Script

1. **Prepare the `paths.yaml` File**:
   - Update the `paths.yaml` file with the paths to your cleaned workflow file and the directory where the processed workflow files will be saved. Here is an example:
     ```yaml
     cleaned_workflow_file: "/path/to/your/cleaned/workflow.txt"
     processed_workflow_dir: "/path/to/your/processed_workflow"
     ```

2. **Run the Script**:
   - Execute the script to process the cleaned workflow log file and create the workflow tree. The script will read the paths from the `paths.yaml` file, process the log file, and save the processed data to the specified output directory.
   - You can run the script using the following command:
     ```sh
     python create_tree.py
     ```

#### Script Overview

1. **process_cleaned_log_file_to_tree(input_path: pathlib.Path, output_dir: pathlib.Path) -> None**:
   - Processes the cleaned log file to create the tree of the workflow.
   - **Parameters**:
     - `input_path` (pathlib.Path): The path to the input log file.
     - `output_dir` (pathlib.Path): The path to the output directory.
   - **Functionality**:
     - Reads the cleaned log file.
     - Extracts shared states, remote functions, iteration blocks, and function blocks.
     - Saves the processed data as pickle files in the specified output directory.

2. **main()**:
   - Creates the tree of the workflow by loading paths from a YAML file, processing the cleaned log file, and saving the processed data to the specified output directory.

By following these steps, you can effectively create a tree representation of your workflow, making it easier to analyze and understand the structure and relationships within the workflow.


### Step 3. How to Use the Workflow Graph and Table Creation Script

This script processes shared states and function blocks to create workflow graphs and tables. It generates visual representations of the workflow and detailed tables summarizing the data, saving them in specified output directories.


#### Steps to Use the Script

1. **Prepare the `paths.yaml` and `config.yaml` Files**:
   - Create a `paths.yaml` file in the root directory of your project and update it with the paths to your output directory and processed workflow directory. Here is an example:
     ```yaml
     output_dir: "/path/to/your/output"
     processed_workflow_dir: "/path/to/your/processed_workflow"
     ```
   - Create a `config.yaml` file in the root directory of your project and update it with the configuration for the graphs to build. Here is an example:
     ```yaml
     graphs_to_build:
       - [function_block_name_1, max_depth_1, rank_1, flatten_first_depth_1]
       - [function_block_name_2, max_depth_2, rank_2, flatten_first_depth_2]
     ```

2. **Run the Script**:
   - Execute the script to process the shared states and function blocks, and create the workflow graphs and tables. The script will read the paths from the `paths.yaml` file and the configuration from the `config.yaml` file, process the data, and save the generated graphs and tables to the specified output directory.
   - You can run the script using the following command:
     ```sh
     python create_workflow_graph_and_tables.py
     ```

#### Script Overview

1. **get_workflow_name(max_depth: int | None, function_block_name: str | None, rank: int = 0) -> str**:
   - Generates a name for the workflow based on the provided parameters.
   - **Parameters**:
     - `max_depth` (int | None): The maximum depth to plot.
     - `function_block_name` (str | None): The name of the function block to plot.
     - `rank` (int): The rank to search for.
   - **Returns**: A string representing the name of the workflow.

2. **create_workflow_graph_and_tables(shared_states: dict[int, SharedState], function_blocks: list[FunctionBlock], output_directory: str | Path, max_depth: int | None = None, function_block_name: str | None = None, rank: int = 0, flatten_first_depth: bool = True) -> None**:
   - Creates workflow graphs and tables from shared states and function blocks.
   - **Parameters**:
     - `shared_states` (dict[int, SharedState]): A dictionary mapping shared state IDs to SharedState objects.
     - `function_blocks` (list[FunctionBlock]): A list of function blocks.
     - `output_directory` (str | Path): The directory where the output graphs and tables will be saved.
     - `max_depth` (int | None, optional): The maximum depth to plot, by default None.
     - `function_block_name` (str | None, optional): The name of the function block to plot, by default None.
     - `rank` (int, optional): The rank to search for, by default 0.
     - `flatten_first_depth` (bool, optional): Whether to flatten the first depth of the graph, by default True.
   - **Functionality**:
     - Converts the shared states dictionary to a list.
     - Ensures the output directory exists.
     - Creates subdirectories for tables and graphs.
     - Generates workflow graphs with different naming conventions for shared states.
     - Creates a table from the shared states and saves it in CSV and LaTeX formats.

3. **main()**:
   - Creates the graph of the workflow by loading paths from a YAML file, processing the cleaned log file, and saving the processed data to the specified output directory.
   - **Functionality**:
     - Loads the paths from the `paths.yaml` file.
     - Loads the configuration from the `config.yaml` file.
     - Loads the shared states and function blocks from pickle files.
     - Creates the workflow graphs and tables based on the configuration.

By following these steps, you can effectively create visual representations and detailed tables of your workflow, making it easier to analyze and understand the structure and relationships within the workflow.
