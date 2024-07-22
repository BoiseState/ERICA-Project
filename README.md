# Synthetic Data Generation for Voter Database

## Overview
This project is designed to generate a synthetic dataset that mirrors the structure and statistical properties of a real voter database. The synthetic data generation process utilizes various Python libraries to handle data manipulation and generation tasks, ensuring the preservation of general patterns and distributions from the original data without compromising personal information.

## Requirements
- Python 3.x
- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations.
- Faker: For generating fake data like addresses and names.
- Pickle: For loading pre-processed name data.

## Data Preparation
Pre-processed data files are required for names and surnames which are loaded and processed at the beginning of the script. These include:
- `male_first_names.pkl`
- `female_first_names.pkl`
- `total_surname.pkl`

These files should contain lists of names that are pre-cleaned and serialized using pickle.

## Script Workflow
1. **Loading and Preprocessing Names:** Loads male and female first names and surnames from pickle files, converting them to uppercase and shuffling.
2. **Loading the Real Dataset:** The real voter file is loaded, and missing values for `Gender`, `FirstName`, and `LastName` are filled with 'Missing'.
3. **Mapping Real to Synthetic Names:** Based on the gender and name, a synthetic name is selected and mapped, ensuring that each real name corresponds to a unique synthetic name.
4. **Data Augmentation:** Additional voter attributes such as `PartyDesc`, `ResCityDesc`, `ResCountyDesc`, `ResZip5`, and `ResState` are generated based on the distributions observed in the original data.
5. **Generating IDs:** Complex attributes like `VoterID`, `LAST4SSN`, and `DriverLicCard` are generated using frequency analysis of digits for each position from the real data to maintain their statistical properties.
6. **Exporting Data:** The synthetic data is saved into a CSV file, preserving the format and structure necessary for further use or analysis.

## Usage
To run the script, ensure all prerequisite libraries are installed and execute the Python script in your preferred environment. The script reads the specified input data file, processes it, and outputs a CSV file containing the synthetic data.

```bash
python generate_synthetic_data.py
