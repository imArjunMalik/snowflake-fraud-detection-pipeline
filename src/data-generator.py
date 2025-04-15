import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for data generation
fake = Faker()

# Define the total number of records
total_records = 10000

# Proportions of each category
rejected_percentage = 0.02
successful_percentage = 0.50
in_review_percentage = 0.48

# Calculate number of records for each category
num_rejected = int(total_records * rejected_percentage)
num_successful = int(total_records * successful_percentage)
num_in_review = total_records - num_rejected - num_successful

# Helper function to generate a synthetic data record
def generate_record(attempt_status, ssn=None, dob=None, first_name=None, last_name=None, is_fraudster=False, fraud_attempt_number=0, base_time=None):
    if ssn is None:
        ssn = fake.ssn()
    if dob is None:
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    if first_name is None:
        first_name = fake.first_name()
    if last_name is None:
        last_name = fake.last_name()
    
    email_domain = random.choice(["@gmail.com", "@yahoo.com", "@hotmail.com", "@protonmail.com", "@ymail.com"])
    email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}{email_domain}"  # Adding randomness to the email
    phone_number = ''.join(random.choices('0123456789', k=10))
    location = fake.address()
    ip_location = location if random.random() > 0.2 else fake.address()  # 20% chance of mismatched location
    use_vpn = random.random() < 0.2  # 20% chance of VPN use
    ip_address = fake.ipv4_private() if use_vpn else fake.ipv4_public()
    
    # Randomly assign device information
    device_info = random.choice(["Windows", "Mac", "Linux"])
    
    if is_fraudster and fraud_attempt_number > 0:
        # Modify the email and phone for fraudulent attempts
        email = f"{first_name.lower()}{random.randint(1, 99)}{email_domain}"
        phone_number = ''.join(random.choices('0123456789', k=10))

    # Timestamps handling: set the timestamp based on fraud_attempt_number and within 72-hour window
    if base_time is None:
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    timestamp = base_time + timedelta(hours=random.uniform(0, 72))  # Randomly space attempts within 72 hours

    # Set label: 1 for fraudulent, 0 for genuine
    label = 1 if is_fraudster else 0

    return {
        "SSN": ssn,
        "DOB": dob,
        "First_Name": first_name,
        "Last_Name": last_name,
        "Email_Address": email,
        "Phone_Number": phone_number,
        "Location": location,
        "IP_Address": ip_address,
        "Device_Info": device_info,
        "IP_Location": ip_location,
        "Use_VPN": use_vpn,
        "Attempt_Status": attempt_status,
        "Timestamp": timestamp,
        "Label": label
    }

# Generate genuine successful records
successful_records = [generate_record("Successful") for _ in range(num_successful)]

# Generate rejected records (considered genuine for this example)
rejected_records = [generate_record("Rejected") for _ in range(num_rejected)]

# Generate fraudulent records with specific distribution of attempts
fraudulent_records = []
fraud_attempts_distribution = {
    "0-5": int(0.25 * num_in_review),
    "6-10": int(0.50 * num_in_review),
    "11-20": int(0.15 * num_in_review),
    "20+": int(0.10 * num_in_review)
}

# 0-5 attempts (25%)
for _ in range(fraud_attempts_distribution["0-5"]):
    ssn = fake.ssn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    fraud_base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    
    # Generate 2-5 fraudulent attempts
    num_attempts = random.randint(2, 5)
    fraud_attempts = [
        generate_record("In Review", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                        is_fraudster=True, fraud_attempt_number=i, base_time=fraud_base_time)
        for i in range(num_attempts - 1)
    ]
    successful_attempt = generate_record("Successful", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                                         is_fraudster=True, fraud_attempt_number=num_attempts, base_time=fraud_base_time)
    fraudulent_records.extend(fraud_attempts + [successful_attempt])

# 6-10 attempts (50%)
for _ in range(fraud_attempts_distribution["6-10"]):
    ssn = fake.ssn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    fraud_base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    
    # Generate 6-10 fraudulent attempts
    num_attempts = random.randint(6, 10)
    fraud_attempts = [
        generate_record("In Review", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                        is_fraudster=True, fraud_attempt_number=i, base_time=fraud_base_time)
        for i in range(num_attempts - 1)
    ]
    successful_attempt = generate_record("Successful", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                                         is_fraudster=True, fraud_attempt_number=num_attempts, base_time=fraud_base_time)
    fraudulent_records.extend(fraud_attempts + [successful_attempt])

# 11-20 attempts (15%)
for _ in range(fraud_attempts_distribution["11-20"]):
    ssn = fake.ssn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    fraud_base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    
    # Generate 11-20 fraudulent attempts
    num_attempts = random.randint(11, 20)
    fraud_attempts = [
        generate_record("In Review", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                        is_fraudster=True, fraud_attempt_number=i, base_time=fraud_base_time)
        for i in range(num_attempts - 1)
    ]
    successful_attempt = generate_record("Successful", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                                         is_fraudster=True, fraud_attempt_number=num_attempts, base_time=fraud_base_time)
    fraudulent_records.extend(fraud_attempts + [successful_attempt])

# 20+ attempts (10%)
for _ in range(fraud_attempts_distribution["20+"]):
    ssn = fake.ssn()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    fraud_base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    
    # Generate 20+ fraudulent attempts
    num_attempts = random.randint(21, 25)
    fraud_attempts = [
        generate_record("In Review", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                        is_fraudster=True, fraud_attempt_number=i, base_time=fraud_base_time)
        for i in range(num_attempts - 1)
    ]
    successful_attempt = generate_record("Successful", ssn=ssn, dob=dob, first_name=first_name, last_name=last_name, 
                                         is_fraudster=True, fraud_attempt_number=num_attempts, base_time=fraud_base_time)
    fraudulent_records.extend(fraud_attempts + [successful_attempt])

# Combine all records (including genuine and fraudulent)
all_records = successful_records + rejected_records + fraudulent_records

# Shuffle the dataset to mix records randomly
random.shuffle(all_records)

# Convert to a DataFrame
df = pd.DataFrame(all_records)

# Save to CSV
df.to_csv('synthetic_account_opening_data.csv', index=False)

# Display the first few rows (optional)
print(df.head())
