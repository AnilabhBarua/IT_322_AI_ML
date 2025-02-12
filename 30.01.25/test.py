import pandas as pd

#Taking user input for dictionary
n = int(input("Enter the number of students: "))

names = []
rolls = []

for i in range(n):
    name = input(f"Enter name of student {i+1}: ")
    roll = input(f"Enter roll number of student {i+1}: ")
    names.append(name)
    rolls.append(roll)

# Create dictionary from user input
data = {'Name': names, 'Roll': rolls}


# Convert dictionary to DataFrame
data = pd.DataFrame(data)

#Print the first few rows
print(data.head(3))  

#basic statistics
print(data.describe())

#access a specific column
print(data['Name'])

#add a new column
data['Marks']=[80,90,95,84,90,89,87]
print(data)

df.to_csv("1.csv", index=True)


