import pandas as pd
import numpy as np
import random
import pickle
from faker import Faker

fake = Faker()

# Load and process pre-processed name data
with open('Pre-processed-names/male_first_names.pkl', 'rb') as file:
    male_first_names = pickle.load(file)

with open('Pre-processed-names/female_first_names.pkl', 'rb') as file:
    female_first_names = pickle.load(file)
    
with open('Pre-processed-names/total_surname.pkl', 'rb') as file:
    total_surnames = pickle.load(file)

# Normalize names to uppercase and shuffle lists
male_first_names = [name.upper() for name in male_first_names if isinstance(name, str)]
female_first_names = [name.upper() for name in female_first_names if isinstance(name, str)]
total_surnames = [surname.upper() for surname in total_surnames if isinstance(surname, str)]

random.shuffle(male_first_names)
random.shuffle(female_first_names)
random.shuffle(total_surnames)

# Load real dataset and fill missing values
data = pd.read_csv("ElectionVoterFile.csv", low_memory=False)
data.fillna({'Gender': 'Missing', 'FirstName': 'Missing', 'LastName': 'Missing'}, inplace=True)
grouped = data.groupby(['Gender', 'FirstName', 'LastName']).size().reset_index(name='Counts')

# Initialize mappings for first and last names
first_name_mapping = {}
last_name_mapping = {}

# Generate mappings from real names to synthetic names
for _, row in grouped.iterrows():
    gender, first_name, last_name = row['Gender'], row['FirstName'], row['LastName']
    if (gender, first_name) not in first_name_mapping:
        if first_name == 'Missing':
            synthetic_first_name = 'Missing'
        else:
            if gender == 'M':
                synthetic_first_name = male_first_names.pop(0)
            elif gender == 'F':
                synthetic_first_name = female_first_names.pop(0)
            else:
                selected_list = random.choice([male_first_names, female_first_names])
                synthetic_first_name = selected_list.pop(0)
        first_name_mapping[(gender, first_name)] = synthetic_first_name
    else:
        synthetic_first_name = first_name_mapping[(gender, first_name)]
        
    if (gender, last_name) not in last_name_mapping:
        if last_name == 'Missing':
            synthetic_last_name = 'Missing'
        else:
            if not total_surnames:
                print("Error: total_surnames list is empty!")
            synthetic_last_name = total_surnames.pop(0)
        last_name_mapping[(gender, last_name)] = synthetic_last_name
    else:
        synthetic_last_name = last_name_mapping[(gender, last_name)]

# Define synthetic dataset
synthetic_data = []
for _, row in data.iterrows():
    gender, first_name, last_name = row['Gender'], row['FirstName'], row['LastName']
    synthetic_first_name = first_name_mapping[(gender, first_name)]
    synthetic_last_name = last_name_mapping[(gender, last_name)]
        
    synthetic_data.append({
        'Gender': None if gender == 'Missing' else gender,
        'FirstName': None if synthetic_first_name == 'Missing' else synthetic_first_name,
        'LastName': None if synthetic_last_name == 'Missing' else synthetic_last_name
    })

synthetic_df = pd.DataFrame(synthetic_data)

num_voters = data.shape[0]
# Simulate PartyDesc based on real data distributions
party_distribution = data['PartyDesc'].value_counts(normalize=True)

party_categories = ['Party-' + str(i+1) for i in range(len(party_distribution))]
party_probs = party_distribution.values.tolist()

PartyDesc_S = np.random.choice(party_categories, num_voters, p=party_probs)
synthetic_df['PartyDesc'] = PartyDesc_S

missing_count = data['PartyDesc'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'PartyDesc'] = np.nan

# ResCityDesc
city_distribution = data['ResCityDesc'].value_counts(normalize=True)

city_categories = city_distribution.index.tolist()
city_probs = city_distribution.values.tolist()

ResCityDesc_S = np.random.choice(city_categories, num_voters, p=city_probs)
synthetic_df['ResCityDesc'] = ResCityDesc_S

missing_count = data['ResCityDesc'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'ResCityDesc'] = np.nan
    

# ResCountyDesc
county_distribution = data['ResCountyDesc'].value_counts(normalize=True)

county_categories = county_distribution.index.tolist()
county_probs = county_distribution.values.tolist()

ResCountyDesc_S = np.random.choice(county_categories, num_voters, p=county_probs)
synthetic_df['ResCountyDesc'] = ResCountyDesc_S

missing_count = data['ResCountyDesc'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'ResCountyDesc'] = np.nan
    
# ResZip5
zip_distribution = data['ResZip5'].value_counts(normalize=True)

zip_categories = zip_distribution.index.tolist()
zip_probs = zip_distribution.values.tolist()

ResZip5_S = np.random.choice(zip_categories, num_voters, p=zip_probs)
synthetic_df['ResZip5'] = ResZip5_S

missing_count = data['ResZip5'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'ResZip5'] = np.nan
    

# ResState
state_S = ['ID' for _ in range(num_voters)]
synthetic_df['ResState'] = state_S

missing_count = data['ResState'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'ResState'] = np.nan

    
# BirthDate
data['BirthDate'] = pd.to_datetime(data['BirthDate'], format='%d/%m/%Y', errors='coerce')
data['BirthYear'] = data['BirthDate'].dt.year
year_distribution = data['BirthYear'].value_counts(normalize=True)
synthetic_birth_years = np.random.choice(year_distribution.index, size=num_voters, p=year_distribution.values)
synthetic_df['BirthYear'] = synthetic_birth_years
# Making birth years are integers
synthetic_df['BirthYear'] = [int(year) for year in synthetic_birth_years]



# Generate synthetic residential street addresses
num_unique = data['ResStreetAddress'].nunique()
num_repetitive = num_voters - num_unique
unique_addresses = [fake.street_address().upper() for _ in range(num_unique)]
repetitive_addresses = np.random.choice(unique_addresses, size=num_repetitive, replace=True)
all_addresses = list(unique_addresses) + list(repetitive_addresses)
random.shuffle(all_addresses)
synthetic_df['ResStreetAddress'] = all_addresses


# Function to analyze the frequency of digits in Voter IDs
def voter_id_frequency_analysis(voter_id_column):
    """Analyzes frequency of each digit at each position in Voter IDs."""
    max_length = max(len(str(voter_id)) for voter_id in voter_id_column)

    # Dictionary to collect digit frequencies at each position
    digit_lists = {i: [] for i in range(max_length)}

    for voter_id in voter_id_column:
        voter_id_str = str(voter_id)
        length = len(voter_id_str)

        # Collect digits for each position
        for i in range(length):
            digit = voter_id_str[i]
            digit_lists[i].append(int(digit))

    # Calculate and store frequency of each digit at each position
    digit_frequencies = []
    for i in range(max_length):
        if digit_lists[i]:
            frequency_series = pd.Series(digit_lists[i])
            frequency_count = frequency_series.value_counts(normalize=True)
            digit_frequencies.append(frequency_count)
        else:
            digit_frequencies.append(pd.Series([]))

    return digit_frequencies

# Generate synthetic Voter IDs based on analyzed frequencies
def generate_synthetic_voter_ids(digit_frequencies, num_ids, id_length=9):
    """Generates synthetic Voter IDs using the digit frequencies."""
    synthetic_voter_ids = []
    
    for _ in range(num_ids):
        new_id = []
        # Generate each digit of the ID based on its position-specific frequency
        for position in range(id_length):
            if position < len(digit_frequencies):
                current_freq = digit_frequencies[position]
                digits = current_freq.index.tolist()
                probabilities = current_freq.values
                chosen_digit = np.random.choice(digits, p=probabilities)
                new_id.append(str(chosen_digit))
            else:
                new_id.append(str(np.random.randint(0, 10)))
        synthetic_voter_ids.append(''.join(new_id))
    return synthetic_voter_ids

# Perform frequency analysis on real Voter IDs
frequency_analysis_result = voter_id_frequency_analysis(data['VoterID'])
synthetic_voter_ids = generate_synthetic_voter_ids(frequency_analysis_result, num_voters)

# Add synthetic Voter IDs to DataFrame
synthetic_df['VoterID'] = synthetic_voter_ids

# Analyze and generate synthetic SSNs (last 4 digits)
def last4ssn_frequency_analysis(last4ssn_column):
    """Analyzes frequency of each digit in the last 4 digits of SSNs."""
    digit_lists = {i: [] for i in range(4)}

    for ssn in last4ssn_column.dropna():
        ssn_str = str(ssn).strip()
        if ssn_str.isdigit() and len(ssn_str) == 4:
            for i in range(len(ssn_str)):
                digit = ssn_str[i]
                digit_lists[i].append(int(digit))

    digit_frequencies = []
    for i in range(4):
        if digit_lists[i]:
            frequency_series = pd.Series(digit_lists[i])
            frequency_count = frequency_series.value_counts(normalize=True)
            digit_frequencies.append(frequency_count)
        else:
            digit_frequencies.append(pd.Series([]))

    return digit_frequencies

def generate_synthetic_last4ssn(digit_frequencies, num_ids):
    """Generates synthetic last 4 digits of SSNs based on analyzed frequencies."""
    synthetic_last4ssn = []
    
    for _ in range(num_ids):
        new_last4ssn = []
        for position in range(4):
            current_freq = digit_frequencies[position]
            digits = current_freq.index.tolist()
            probabilities = current_freq.values
            chosen_digit = np.random.choice(digits, p=probabilities)
            new_last4ssn.append(str(chosen_digit))
        synthetic_last4ssn.append(''.join(new_last4ssn))
    return synthetic_last4ssn

frequency_analysis_result = last4ssn_frequency_analysis(data['LAST4SSN'])
synthetic_last4ssn = generate_synthetic_last4ssn(frequency_analysis_result, num_voters)
synthetic_df['LAST4SSN'] = synthetic_last4ssn

# Analyze and generate synthetic Driver's License Cards
def driverliccard_frequency_analysis(driverliccard_column):
    """Analyzes frequency of each character in Driver's License Cards."""
    alpha_lists = {0: [], 1: [], 8: []}  # Positions for alphabetic characters
    digit_lists = {i: [] for i in range(2, 8)}  # Positions for numeric characters

    for card in driverliccard_column.dropna():
        card_str = str(card).strip()
        if len(card_str) >= 9:
            card_str = card_str[:9]

            # Collect alphabetic characters at specific positions
            if card_str[0].isalpha():
                alpha_lists[0].append(card_str[0])
            if card_str[1].isalpha():
                alpha_lists[1].append(card_str[1])
            if card_str[-1].isalpha():
                alpha_lists[8].append(card_str[-1])
            
            # Collect numeric characters at remaining positions
            for i in range(2, 8):
                if card_str[i].isdigit():
                    digit_lists[i].append(int(card_str[i]))

    alpha_frequencies = {i: pd.Series(lst).value_counts(normalize=True) if lst else pd.Series([])
                         for i, lst in alpha_lists.items()}
    digit_frequencies = {i: pd.Series(lst).value_counts(normalize=True) if lst else pd.Series([])
                         for i, lst in digit_lists.items()}

    return alpha_frequencies, digit_frequencies

def generate_synthetic_driverliccard(alpha_frequencies, digit_frequencies, num_ids):
    """Generates synthetic Driver's License Cards using the analyzed frequencies."""
    synthetic_driverliccards = []

    for _ in range(num_ids):
        new_card = []
        # Alphabetic characters for positions 0, 1, and last
        for position in [0, 1]:
            current_freq = alpha_frequencies[position]
            characters = current_freq.index.tolist()
            probabilities = current_freq.values
            chosen_character = np.random.choice(characters, p=probabilities)
            new_card.append(chosen_character)

        # Numeric characters for positions 2 to 7
        for position in range(2, 8):
            current_freq = digit_frequencies[position]
            digits = current_freq.index.tolist()
            probabilities = current_freq.values
            chosen_digit = np.random.choice(digits, p=probabilities)
            new_card.append(str(chosen_digit))
        
        # Alphabetic character for the last position
        current_freq_8 = alpha_frequencies[8]
        characters_8 = current_freq_8.index.tolist()
        probabilities_8 = current_freq_8.values
        chosen_character_8 = np.random.choice(characters_8, p=probabilities_8)
        new_card.append(chosen_character_8)
        
        synthetic_driverliccards.append(''.join(new_card))
    return synthetic_driverliccards

alpha_frequencies, digit_frequencies = driverliccard_frequency_analysis(data['DriverLicCard'])
synthetic_driverliccards = generate_synthetic_driverliccard(alpha_frequencies, digit_frequencies, num_voters)
synthetic_df['DriverLicCard'] = synthetic_driverliccards

# Prepare to handle missing values in synthetic data, similar to real data
used_indices_dln = set()
missing_count = data['DriverLicCard'].isna().sum()
if missing_count > 0:
    missing_indices = np.random.choice(synthetic_df.index, missing_count, replace=False)
    synthetic_df.loc[missing_indices, 'DriverLicCard'] = np.nan  # Set missing values as in the original data
    used_indices_dln.update(missing_indices)  # Keep track of indices used for setting missing values

# Add the synthetic attributes to the synthetic dataset
synthetic_df['DriverLicCard'].astype(str).apply(len).value_counts()

# Analyze the lengths of real Driver's License Cards to adjust synthetic cards accordingly
lengths = data['DriverLicCard'].astype(str).apply(len)
length_counts = lengths.value_counts()

for length, count in length_counts.items():
    if length == 3 or length == 9:
        continue  # Skip adjustments for standard lengths

    available_indices = synthetic_df.index.difference(list(used_indices_dln))
    random_indices = np.random.choice(available_indices, size=count, replace=False)

    # Adjust the length of DriverLicCard entries
    if 1 <= length < 9:
        # Shorten synthetic DriverLicCard entries to match found lengths less than 9
        synthetic_df.loc[random_indices, 'DriverLicCard'] = synthetic_df.loc[random_indices, 'DriverLicCard'].apply(
            lambda x: x[:length] if isinstance(x, str) and len(x) >= length else x
        )
    elif length == 10:
        # Adjust entries to length of 10 by inserting a random digit at position 5
        synthetic_df.loc[random_indices, 'DriverLicCard'] = synthetic_df.loc[random_indices, 'DriverLicCard'].apply(
            lambda x: x[:5] + str(np.random.randint(0, 10)) + x[5:] if isinstance(x, str) and len(x) >= 9 else x
        )

    used_indices_dln.update(random_indices)  # Update the set of used indices to avoid reusing them


# List of columns in the desired order
desired_order = ['VoterID', 'BirthYear', 'LAST4SSN', 'DriverLicCard',
                 'FirstName', 'LastName', 'Gender', 'ResStreetAddress', 
                 'ResCityDesc', 'ResState', 'ResZip5', 'ResCountyDesc', 
                 'PartyDesc']

# Reorder the columns of the synthetic data DataFrame
synthetic_df = synthetic_df[desired_order]

synthetic_df = synthetic_df.copy()
synthetic_df.replace(['nan', 'NaN'], np.nan, inplace=True)

# Save the DataFrame to a CSV file
synthetic_df.to_csv('Synthetic-Data/synthetic_data_2.csv', index=False)
